import json
import torch
import os
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# ================= 🔴 配置区域 🔴 =================
# 1. 待评测的模型路径
#   - 基线: "/mnt/raid/hss/model/Qwen3-VL-8B-Instruct"
#   - 微调: "/home/houshuoshuo/models/SilkRoad-MMT-8B" (合并后的路径)
MODEL_PATH = "/mnt/raid/hss/model/SilkRoad-MMT-8B"

# 2. 测试集文件夹 (读取原始分语言文件)
TEST_DIR = "/home/houshuoshuo/qlora_data/split/test"

# 3. 结果保存文件名
OUTPUT_FILE = "pred_finetuned_v2_3epoch.json"

# 4. ✅ 新增：图片根目录 (必须配置，否则找不到图片)
IMAGE_ROOT = "/mnt/raid/hss/dataset/Image50K"


# ===================================================

def run_inference():
    print(f"⏳ Loading Model from: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # 语言映射表 (用于构造 Prompt)
    lang_map = {
        'ug': 'Uyghur', 'uyghur': 'Uyghur',
        'uz': 'Uzbek', 'uzbek': 'Uzbek',
        'kk': 'Kazakh', 'kazakh': 'Kazakh',
        'ur': 'Urdu', 'urdu': 'Urdu',
        'ky': 'Kyrgyz', 'kyrgyz': 'Kyrgyz',
        'tg': 'Tajik', 'tajik': 'Tajik'
    }

    files = [f for f in os.listdir(TEST_DIR) if f.endswith('.json')]
    files.sort()
    print(f"🌍 Found {len(files)} language test files: {files}")

    all_results = []

    for filename in files:
        # 提取语言代码 (如 'kazakh')
        lang_code = filename.replace('.json', '').lower()

        # 处理可能的 dataset_ 前缀
        if 'dataset_' in lang_code:
            lang_code = lang_code.split('_')[1]

        target_lang_name = lang_map.get(lang_code, lang_code.capitalize())

        file_path = os.path.join(TEST_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"🚀 Processing [{lang_code}] -> Target: {target_lang_name} ({len(data)} samples)...")

        for item in tqdm(data, desc=lang_code):
            # ✅ 适配原始数据格式
            # 原始格式通常是: {"path": "xxx.jpg", "src_text": "...", "tgt_text": "..."}

            rel_path = item.get('path', '')
            img_filename = os.path.basename(rel_path)
            # 拼接绝对路径
            image_path = os.path.join(IMAGE_ROOT, img_filename)

            src_text = item.get('src_text', '')
            ground_truth = item.get('tgt_text', '')

            # ✅ 现场构造 Prompt (必须与训练时完全一致！)
            user_prompt = f"Please translate the description of this image into {target_lang_name}.\nEnglish Source: {src_text}"

            try:
                image = Image.open(image_path).convert("RGB")

                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]
                    }
                ]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                all_results.append({
                    "language": lang_code,
                    "image_path": image_path,
                    "src": user_prompt,  # 记录完整的 Prompt 方便查看
                    "ref": ground_truth,
                    "hyp": output_text
                })

            except Exception as e:
                # print(f"   ❌ Error on {image_path}: {e}")
                pass

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"✅ All predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_inference()
