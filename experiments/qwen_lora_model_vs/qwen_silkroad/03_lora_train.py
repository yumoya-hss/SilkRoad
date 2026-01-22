import os

# ✅ 1. 显存碎片优化 (保持不变)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from PIL import Image

import transformers
from transformers import (
    AutoModelForImageTextToText,  # 使用新版类名
    AutoProcessor,
    Trainer,
    TrainingArguments
)
# ✅ 2. 引入检查点工具
from transformers.trainer_utils import get_last_checkpoint

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# ================= 🔴 核心配置区域 🔴 =================
MODEL_ID = "models/Qwen3-VL-8B-Instruct"
DATA_PATH = "outputs/silkroad_train.json"
OUTPUT_DIR = "outputs/qwen_native_output"

# H100 配置
BATCH_SIZE = 8
GRAD_ACCUM = 8
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_LEN = 2048

# 分辨率限制
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28

# ✅ 3. 新增：保存策略 (防止训练几小时后崩溃白跑)
SAVE_STEPS = 500  # 每跑 500 步存一次档
SAVE_TOTAL_LIMIT = 2  # 只保留最近的 2 个存档，节省硬盘空间


# =======================================================

def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ✅ 4. 自定义 Dataset (保持不变)
class QwenVLDataset(torch.utils.data.Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image_path = item['images'][0]
        conversations = item['conversations']

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"❌ Bad Image: {image_path} | {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        user_text = conversations[0]['value'].replace('<image>', '').strip()
        assistant_text = conversations[1]['value']

        messages_full = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
        ]
        messages_prompt = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}
        ]

        text_full = self.processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
        text_prompt = self.processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[text_full], images=[image], videos=None, padding=False, return_tensors="pt")
        inputs_prompt = self.processor(text=[text_prompt], images=[image], padding=False, return_tensors="pt")

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()
        prompt_len = inputs_prompt["input_ids"].shape[1]

        if prompt_len < len(labels):
            labels[:prompt_len] = -100
        else:
            labels[:len(labels) - 1] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"][0],
            "labels": labels
        }


# ✅ 5. 自定义 Collator (保持不变)
@dataclass
class DataCollatorForQwenVL:
    processor: AutoProcessor

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        pixel_values = [f["pixel_values"] for f in features]
        image_grid_thw = [f["image_grid_thw"] for f in features]
        labels = [f["labels"] for f in features]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                           padding_value=self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        pixel_values_cat = torch.cat(pixel_values, dim=0)
        image_grid_thw_cat = torch.stack(image_grid_thw, dim=0)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "pixel_values": pixel_values_cat,
            "image_grid_thw": image_grid_thw_cat,
            "labels": labels_padded
        }


# ✅ 6. 核心修改：防崩溃 Trainer
class RobustTrainer(Trainer):
    """
    一个强壮的 Trainer，遇到 OOM 错误时不会崩溃，而是跳过该 batch 继续训练。
    """

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            # 尝试正常执行训练步
            return super().training_step(model, inputs, num_items_in_batch)
        except torch.cuda.OutOfMemoryError:
            # 捕获 OOM 错误
            torch.cuda.empty_cache()
            print(
                f"\n⚠️ [OOM Warning] Step {self.state.global_step}: GPU Out of Memory! Skipping this batch to continue training...")
            # 返回一个 0 损失，不影响梯度，但能保持训练循环不中断
            return torch.tensor(0.0, device=model.device, requires_grad=True)


def train():
    print("⏳ Loading Processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )

    print("⏳ Loading Model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        ),
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=64, lora_alpha=128, target_modules="all-linear",
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", use_dora=True,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("⏳ Processing Data...")
    raw_data = load_data(DATA_PATH)
    dataset = QwenVLDataset(raw_data, processor)
    collator = DataCollatorForQwenVL(processor)

    # ✅ 7. 自动检查是否存在 Checkpoint
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

    if last_checkpoint:
        print(f"🔄 发现存档，将从断点继续训练: {last_checkpoint}")
    else:
        print("🚀 未发现存档，开始新训练...")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_torch_fused",
        bf16=True,
        logging_steps=5,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        report_to="none",
        remove_unused_columns=False,

        # ✅ 8. 修改保存策略：按步数保存
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,  # 只保留最近2个，防止硬盘爆满

        # 确保继续训练时能加载之前的状态
        overwrite_output_dir=False,
    )

    # ✅ 9. 使用自定义的 RobustTrainer
    trainer = RobustTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("🚀 Starting Robust Training...")

    # ✅ 10. 启动训练 (传入断点路径)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print("💾 Saving Final Model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    train()
