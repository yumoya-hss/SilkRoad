import json
import sys
import os
import torch
import argparse
import gc
import re
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)

# ==========================================
# 0. 用户配置
# ==========================================
QWEN_PATH = "/mnt/raid/zsb/llm_models/Qwen3-32B"

# ⚡ 批处理大小
# 建议：24G显存(4bit)设为16；48G+显存(FP16)设为32
# 如果显存不够，可以改小到 8 或 4
BATCH_SIZE = 16 

# ==========================================
# 1. 语言配置与脚本约束
# ==========================================
VALID_LANG_KEYS = {
    "uzbek", "kazakh", "kyrgyz", "tajik",
    "urdu", "bengali", "pashto", "hindi", "nepali", "marathi", "telugu", "tamil",
    "vietnamese", "thai", "indonesian", "khmer", "lao", "burmese", "malay",
    "persian", "arabic", "turkish", "hebrew",
    "swahili", "yoruba", "zulu", "amharic", "hausa",
    "uyghur", "mongolian", "korean", "japanese", "chinese"
}

QWEN_LANG_SCRIPT_MAP = {
    "uyghur": "Uyghur (in standard Arabic script/UEY)",
    "kazakh": "Kazakh (in Cyrillic script)",
    "uzbek": "Uzbek (in Latin script)",
    "kyrgyz": "Kyrgyz (in Cyrillic script)",
    "tajik": "Tajik (in Cyrillic script)",
    "urdu": "Urdu (in Arabic script)",
    "chinese": "Chinese (Simplified)",
    "mongolian": "Mongolian (in Cyrillic script)",
}

# ==========================================
# 2. 工具函数
# ==========================================
def load_data(file_path):
    print(f"📖 读取数据: {file_path} ...")
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        sys.exit(1)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content: return []
            if content.startswith('['):
                return json.loads(content)
            else:
                return [json.loads(line) for line in content.split('\n') if line.strip()]
    except Exception as e:
        print(f"❌ 数据错误: {e}"); sys.exit(1)

def save_data(data, output_file):
    """保存数据到磁盘"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ 保存失败: {e}")

def clean_translation(text):
    """
    清洗函数：去除 <think> 标签、Markdown 和常见前缀
    """
    if not text: return ""
    # 1. 去除 <think>...内容...</think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 2. 去除 Markdown
    text = text.replace('```', '').strip()
    # 3. 去除常见前缀
    prefixes = ["Translation:", "Translated:", "Output:", "Answer:", "翻译：", "Result:"]
    for p in prefixes:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()
    # 4. 去除首尾引号
    if text.startswith('"') and text.endswith('"') and len(text) > 2:
        text = text[1:-1]
    return text.strip()

# ==========================================
# 3. 翻译主逻辑
# ==========================================
def run_translation(model, tokenizer, data, lang_key, device, output_file):
    target_lang_desc = QWEN_LANG_SCRIPT_MAP.get(lang_key, lang_key.capitalize())
    
    print(f"\n🚀 开始翻译: {lang_key} -> {target_lang_desc}")
    print("💾 模式: 实时保存 (每批次完成后立即写入JSON)")
    
    # System Prompt: 专家角色 + 严格约束
    sys_prompt = (
        f"You are a professional linguist and native speaker of {target_lang_desc}.\n"
        "### Task\n"
        f"Translate the English image caption into natural, grammatical {target_lang_desc}.\n\n"
        "### Strict Rules\n"
        "1. **Script Compliance**: Use the OFFICIAL script ONLY (e.g., Arabic for Uyghur). Do NOT use transliteration.\n"
        "2. **Accuracy**: Preserve the exact meaning without hallucination.\n"
        "3. **No Thinking**: Do NOT output <think> tags.\n"
        "4. **Output**: Output ONLY the translation.\n"
    )

    # 1. 筛选未翻译的数据 (断点续传)
    todo_indices = []
    for idx, item in enumerate(data):
        if 'translations' not in item: item['translations'] = {}
        if lang_key not in item['translations']: item['translations'][lang_key] = {}
        
        existing_s = item['translations'][lang_key].get("short_qwen", "")
        existing_l = item['translations'][lang_key].get("long_qwen", "")
        
        if not (existing_s and existing_l):
            todo_indices.append(idx)

    if not todo_indices:
        print(f"✅ {lang_key} 全部已完成 ({len(data)} 条)，跳过。")
        return

    # 2. 定义批量生成函数
    def generate_batch(texts, max_len):
        valid_map = {i: t for i, t in enumerate(texts) if t.strip()}
        if not valid_map: return [""] * len(texts)
        
        prompts = []
        for txt in valid_map.values():
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": txt}]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

        with torch.inference_mode():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_len + 32,
                temperature=0.0,             # Greedy Decoding (最快)
                do_sample=False,             # 关闭采样
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        gen_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        
        results = [""] * len(texts)
        for i, (v_idx, raw_text) in enumerate(zip(valid_map.keys(), decoded)):
            results[v_idx] = clean_translation(raw_text)
        return results

    # 3. 循环处理
    steps = 0
    try:
        pbar = tqdm(total=len(todo_indices), desc=f"Processing {lang_key}", mininterval=1.0)
        
        for i in range(0, len(todo_indices), BATCH_SIZE):
            batch_idxs = todo_indices[i : i + BATCH_SIZE]
            
            # 提取原文
            short_src = [data[idx].get('short_caption_best', "") for idx in batch_idxs]
            long_src = [data[idx].get('long_caption_best', "") for idx in batch_idxs]

            # 执行翻译
            short_out = generate_batch(short_src, 128)
            long_out = generate_batch(long_src, 256)

            # 回写数据
            for j, data_idx in enumerate(batch_idxs):
                data[data_idx]['translations'][lang_key]['short_qwen'] = short_out[j]
                data[data_idx]['translations'][lang_key]['long_qwen'] = long_out[j]
                
                # [可选] 打印第一条做监控
                if j == 0:
                    tqdm.write(f"📝 Src: {short_src[j][:30]}...  =>  🟢 Tgt: {short_out[j][:30]}...")

            pbar.update(len(batch_idxs))
            steps += 1

            # 🔥 [核心修改] 每一批次处理完，立即保存！
            # 这样你打开 JSON 文件，永远能看到最新的翻译结果
            save_data(data, output_file)
                
    except KeyboardInterrupt:
        print("\n🛑 用户停止！正在最后一次保存...")
        save_data(data, output_file)
        sys.exit(0)

    # 跑完一种语言，再次确保保存
    save_data(data, output_file)

# ==========================================
# 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="原始输入文件")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件 (支持增量)")
    parser.add_argument("--langs", type=str, required=True, help="目标语言列表")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    langs = [l.strip().lower() for l in args.langs.split(',')]
    valid_langs = [l for l in langs if l in VALID_LANG_KEYS]
    if not valid_langs: 
        print("❌ 无有效语言。")
        return

    # 增量加载逻辑
    if os.path.exists(args.output_file):
        print(f"🔄 检测到进度文件: {args.output_file}，加载以继续...")
        data = load_data(args.output_file)
    else:
        print(f"🆕 首次运行，加载原始文件: {args.input_file}")
        data = load_data(args.input_file)

    device = f"cuda:{args.gpu_id}"
    
    print(f"[{device}] Loading Qwen3-32B...")
    try:
        # 标准加载
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH, 
            torch_dtype=torch.float16, 
            device_map=device, 
            trust_remote_code=True
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 逐个语言处理
    for lang in valid_langs:
        run_translation(model, tokenizer, data, lang, device, args.output_file)

    print("\n🎉 全部任务完成！")

if __name__ == "__main__":
    main()