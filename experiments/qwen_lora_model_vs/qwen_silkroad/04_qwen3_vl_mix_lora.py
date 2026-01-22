import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
import os

# ================= 配置 =================
BASE_MODEL_PATH = "models/Qwen3-VL-8B-Instruct"
ADAPTER_PATH = "outputs/qwen_native_output"
SAVE_PATH = "models/SilkRoad-MMT-8B"
# =======================================

print(f"⏳ Loading Base Model from {BASE_MODEL_PATH}...")
# 建议修改：使用 bfloat16 与训练保持一致
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,  # ✅ 改为 bfloat16
    device_map="auto",
    trust_remote_code=True
)

print(f"⏳ Loading LoRA Adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("🔗 Merging...")
model = model.merge_and_unload()

print(f"💾 Saving Merged Model to {SAVE_PATH}...")
model.save_pretrained(SAVE_PATH)

print("💾 Saving Processor...")
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
processor.save_pretrained(SAVE_PATH)

print("✅ Done! Full model is ready.")
