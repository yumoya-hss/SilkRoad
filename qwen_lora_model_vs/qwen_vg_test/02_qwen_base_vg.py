import json
import torch
import os
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# ================= 🔴 配置区域 🔴 =================
# 1. 基线模型路径
MODEL_PATH = "/mnt/raid/hss/model/Qwen3-VL-8B-Instruct"

# 2. 输入数据文件 (包含50条 Multi30k 数据)
INPUT_JSON = "/home/houshuoshuo/qlora_data/test/vg_50_crop.json"

# 3. 结果保存文件名
OUTPUT_FILE = "pred_100_baseline_vg.json"

# 4. 目标语言配置
TARGET_LANGUAGES = {
    'ug': 'Uyghur',
    'uz': 'Uzbek',
    'kk': 'Kazakh',
    'ur': 'Urdu',
    'ky': 'Kyrgyz',
    'tg': 'Tajik'
}
# ===================================================

def generate_response(model, processor, image, prompt_text):
    """
    通用生成函数
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    # 构造输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # 推理
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False  # 贪婪搜索，保证基线结果稳定
        )

    # 解码
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

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

    # 读取输入数据
    if not os.path.exists(INPUT_JSON):
        print(f"❌ Error: Input file not found at {INPUT_JSON}")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📄 Loaded {len(data)} items from {INPUT_JSON}")
    
    all_results = []

    # 遍历每条数据
    for item in tqdm(data, desc="Processing Multi30k"):
        # 1. 获取图片路径 (处理可能存在的 // 问题)
        raw_path = item.get('saved_path', '')
        # 简单清洗路径：把 // 替换为 /
        image_path = raw_path.replace('//', '/')
        
        # 2. 获取源英文文本
        src_en = item.get('src_en', '')
        
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}, skipping...")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            
            # 3. 遍历 6 种目标语言进行翻译
            for lang_code, target_lang_name in TARGET_LANGUAGES.items():
                
                # 构造翻译指令 (Baseline Prompt)
                trans_prompt = f"Please translate the description of this image into {target_lang_name}.\nEnglish Source: {src_en}"
                
                # 执行推理
                hyp_translation = generate_response(
                    model, processor, image, trans_prompt
                )
                
                # 存入结果
                all_results.append({
                    "language": lang_code,
                    "image_id": item.get('image_id'),
                    "image_path": image_path,
                    "src_prompt": trans_prompt, # 记录完整的 prompt
                    "src_en": src_en,           # 原始英文
                    "ref": "",                  # 基线推理无参考答案(或后续填入)
                    "hyp": hyp_translation      # 模型翻译结果
                })

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            continue

    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Finished! Generated {len(all_results)} entries.")
    print(f"   (50 items * 6 langs = 300 expected)")
    print(f"📂 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_inference()
