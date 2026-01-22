import json
import sys
import os
import torch
import argparse
import gc
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM
)

# ==========================================
# 0. 用户配置区域
# ==========================================

# MADLAD-400 (7B-MT) 路径
MADLAD_PATH = "models/madlad400-7b-mt"

# 批处理大小
BATCH_SIZE = 64 

# ==========================================
# 1. 语言映射表 (仅保留 MADLAD 支持的)
# ==========================================
MADLAD_LANG_MAP = {
    # --- 中亚 ---
    "uyghur": "<2ug>", "uzbek": "<2uz>", "kazakh": "<2kk>", "kyrgyz": "<2ky>", "tajik": "<2tg>",
    # --- 南亚 ---
    "urdu": "<2ur>", "bengali": "<2bn>", "pashto": "<2ps>", "hindi": "<2hi>", 
    "nepali": "<2ne>", "marathi": "<2mr>", "telugu": "<2te>", "tamil": "<2ta>",
    # --- 东南亚 ---
    "vietnamese": "<2vi>", "thai": "<2th>", "indonesian": "<2id>", "khmer": "<2km>",
    "lao": "<2lo>", "burmese": "<2my>", "malay": "<2ms>",
    # --- 中东/西亚 ---
    "persian": "<2fa>", "arabic": "<2ar>", "turkish": "<2tr>", "hebrew": "<2he>",
    # --- 非洲 ---
    "swahili": "<2sw>", "yoruba": "<2yo>", "zulu": "<2zu>", "amharic": "<2am>", "hausa": "<2ha>",
    # --- 东亚/其它 ---
    "mongolian": "<2mn>", "korean": "<2ko>", "japanese": "<2ja>", "chinese": "<2zh>",
}

def load_data(file_path):
    print(f"📖 读取数据: {file_path} ...")
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if f.read(1) == '[':
                f.seek(0); data = json.load(f)
            else:
                f.seek(0); [data.append(json.loads(line)) for line in f if line.strip()]
    except Exception as e:
        print(f"❌ 数据错误: {e}"); sys.exit(1)
    return data

# ==========================================
# 2. MADLAD 翻译核心
# ==========================================
def translate_madlad(model, tokenizer, data, lang_key, device):
    """
    专门用于 MADLAD-400 的翻译函数
    """
    madlad_token = MADLAD_LANG_MAP.get(lang_key)
    if not madlad_token:
        print(f"⚠️ MADLAD 暂未配置 {lang_key} 的映射，跳过。")
        return [""] * len(data), [""] * len(data)

    print(f"   >> [MADLAD-400] Target: {lang_key} ({madlad_token}) ...")
    
    short_results, long_results = [], []
    
    # 定义批处理生成函数
    def run_gen(texts, max_len):
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        if not valid_indices: return [""] * len(texts)
        
        # MADLAD 需要在输入前加 target token，例如 "<2zh> I love you"
        inputs_text = [f"{madlad_token} {texts[i]}" for i in valid_indices]

        # 编码
        inputs = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        # 生成配置
        gen_kwargs = {
            "max_new_tokens": max_len,
            "num_beams": 1, # MADLAD 通常推荐 greedy 或少量的 beams
            "do_sample": False
        }
        
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        
        final_res = [""] * len(texts)
        for idx, res in zip(valid_indices, decoded):
            final_res[idx] = res.strip()
        return final_res

    # 分批处理整个数据集
    for i in tqdm(range(0, len(data), BATCH_SIZE), desc=f"   Processing MADLAD"):
        batch = data[i : i + BATCH_SIZE]
        short_src = [item.get('short_caption_best', "") for item in batch]
        long_src = [item.get('long_caption_best', "") for item in batch]

        short_results.extend(run_gen(short_src, 128)) 
        long_results.extend(run_gen(long_src, 256)) 
        
    return short_results, long_results

# ==========================================
# 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--langs", type=str, required=True, help="逗号分隔，例如: uyghur,uzbek")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    # 1. 解析语言
    input_langs = [l.strip().lower() for l in args.langs.split(',')]
    valid_langs = []
    for l in input_langs:
        if l in MADLAD_LANG_MAP:
            valid_langs.append(l)
        else:
            print(f"⚠️ 警告: 语言 '{l}' 不在 MADLAD 支持列表中，已忽略。")
    
    if not valid_langs:
        print("❌ 没有有效的语言需要翻译。")
        return

    device = f"cuda:{args.gpu_id}"
    data = load_data(args.input_file)
    
    # 结果缓存结构
    results_cache = {lang: {} for lang in valid_langs}

    # ----------------------------------------
    # Loading MADLAD-400
    # ----------------------------------------
    print(f"\n[{device}] Loading MADLAD-400 Model...")
    try:
        m_model = AutoModelForSeq2SeqLM.from_pretrained(MADLAD_PATH, torch_dtype=torch.float16).to(device).eval()
        m_tok = AutoTokenizer.from_pretrained(MADLAD_PATH)
        
        for lang in valid_langs:
            s, l = translate_madlad(m_model, m_tok, data, lang, device)
            results_cache[lang]['short'] = s
            results_cache[lang]['long'] = l
            
        del m_model, m_tok; torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        print(f"❌ MADLAD 加载或翻译失败: {e}")
        sys.exit(1)

    # ----------------------------------------
    # 合并保存
    # ----------------------------------------
    print(f"\n🔄 正在写入结果...")
    for idx, item in enumerate(data):
        if 'translations' not in item: item['translations'] = {}
        
        for lang in valid_langs:
            if lang not in item['translations']:
                item['translations'][lang] = {}
            
            # 仅写入 MADLAD 结果
            item['translations'][lang]["short_madlad"] = results_cache[lang]['short'][idx]
            item['translations'][lang]["long_madlad"] = results_cache[lang]['long'][idx]

    print(f"💾 保存至: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("🎉 翻译完成！")

if __name__ == "__main__":
    main()