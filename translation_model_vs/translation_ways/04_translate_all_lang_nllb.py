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
# 0. 用户硬编码配置区域
# ==========================================
# 模型路径 (仅保留 NLLB)
NLLB_PATH = "/mnt/raid/hss/model/nllb-200-3.3B"

# 批处理大小
BATCH_SIZE = 64

# ==========================================
# 1. 语言代码映射表 (NLLB Codes)
# ==========================================
SUPPORTED_LANGS = {
    # --- 中亚 (Central Asia) ---
    "uzbek": "uzn_Latn",      # 乌兹别克语
    "kazakh": "kaz_Cyrl",     # 哈萨克语
    "kyrgyz": "kir_Cyrl",     # 吉尔吉斯语
    "tajik": "tgk_Cyrl",      # 塔吉克语

    # --- 南亚 (South Asia) ---
    "urdu": "urd_Arab",       # 乌尔都语
    "bengali": "ben_Beng",    # 孟加拉语
    "pashto": "pbt_Arab",     # 普什图语
    "hindi": "hin_Deva",      # 印地语
    "nepali": "npi_Deva",     # 尼泊尔语
    "marathi": "mar_Deva",    # 马拉地语
    "telugu": "tel_Telu",     # 泰卢固语
    "tamil": "tam_Taml",      # 泰米尔语

    # --- 东南亚 (Southeast Asia) ---
    "vietnamese": "vie_Latn", # 越南语
    "thai": "tha_Thai",       # 泰语
    "indonesian": "ind_Latn", # 印尼语
    "khmer": "khm_Khmr",      # 高棉语
    "lao": "lao_Laoo",        # 老挝语
    "burmese": "mya_Mymr",    # 缅甸语
    "malay": "zsm_Latn",      # 马来语

    # --- 中东/西亚 (Middle East) ---
    "persian": "pes_Arab",    # 波斯语
    "arabic": "arb_Arab",     # 阿拉伯语
    "turkish": "tur_Latn",    # 土耳其语
    "hebrew": "heb_Hebr",     # 希伯来语

    # --- 非洲 (Africa) ---
    "swahili": "swh_Latn",    # 斯瓦希里语
    "yoruba": "yor_Latn",     # 约鲁巴语
    "zulu": "zul_Latn",       # 祖鲁语
    "amharic": "amh_Ethi",    # 阿姆哈拉语
    "hausa": "hau_Latn",      # 豪萨语

    # --- 东亚/其它 (East Asia / Others) ---
    "uyghur": "uig_Arab",     # 维吾尔语
    "mongolian": "mon_Cyrl",  # 蒙古语
    "korean": "kor_Hang",     # 韩语
    "japanese": "jpn_Jpan",   # 日语
    "chinese": "zho_Hans",    # 中文
}

def clean_text(text, lang_code):
    """清理 NLLB 可能产生的语言代码前缀"""
    if not text: return ""
    text = text.strip()
    if text.startswith(lang_code):
        text = text[len(lang_code):].strip()
    return text

def load_data(file_path):
    print(f"📖 读取数据: {file_path} ...")
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
            else:
                for line in f:
                    if line.strip(): data.append(json.loads(line))
    except Exception as e:
        print(f"❌ 数据错误: {e}")
        sys.exit(1)
    print(f"✅ 成功加载 {len(data)} 条数据")
    return data

# ==========================================
# 2. 翻译核心函数 (仅 NLLB 逻辑)
# ==========================================
def translate_dataset(model, tokenizer, data, target_code, device):
    print(f"   >> [NLLB] Translating to code: {target_code} ...")
    
    short_results = []
    long_results = []
    
    # 获取 NLLB 的 forced_bos_token_id (强制目标语言开头)
    nllb_bos_id = tokenizer.convert_tokens_to_ids(target_code)

    for i in tqdm(range(0, len(data), BATCH_SIZE), desc=f"   Processing"):
        batch = data[i : i + BATCH_SIZE]
        
        # 提取文本
        short_src = [item.get('short_caption_best', item.get('short_caption', "")) for item in batch]
        long_src = [item.get('long_caption_best', item.get('long_caption', "")) for item in batch]
        
        def run_gen(texts, max_new_tokens):
            # 过滤空文本
            valid_indices = [i for i, t in enumerate(texts) if t.strip()]
            if not valid_indices: return [""] * len(texts)
            
            valid_texts = [texts[i] for i in valid_indices]
            # NLLB 输入不需要特殊前缀，直接输入源文本即可，模型会自动处理
            inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_beams": 5,
                "do_sample": False,
                "use_cache": True,
                "forced_bos_token_id": nllb_bos_id  # 关键：指定目标语言
            }
            
            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kwargs)
            
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            
            # 还原顺序 + 清洗
            final_batch_res = [""] * len(texts)
            for idx, res in zip(valid_indices, decoded):
                final_batch_res[idx] = clean_text(res, target_code)
            return final_batch_res
        
        # Short (96 tokens), Long (256 tokens)
        short_results.extend(run_gen(short_src, 96)) 
        long_results.extend(run_gen(long_src, 256)) 
        
    return short_results, long_results

# ==========================================
# 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="输入文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="最终结果保存路径")
    parser.add_argument("--langs", type=str, required=True, help="逗号分隔的语言列表, 例如: uyghur,uzbek,urdu")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    # 1. 解析目标语言
    target_lang_names = [l.strip().lower() for l in args.langs.split(',')]
    valid_langs = []
    
    print("\n🔍 检查语言列表...")
    for lang in target_lang_names:
        if lang not in SUPPORTED_LANGS:
            print(f"❌ 警告: 不支持的语言 '{lang}'，将跳过。")
        else:
            print(f"✅ 待处理: {lang:<12} (Code: {SUPPORTED_LANGS[lang]})")
            valid_langs.append(lang)
            
    if not valid_langs:
        print("没有有效的语言，退出。")
        return

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    data = load_data(args.input_file)

    # 结果缓存: results_cache[lang] = {'short': [], 'long': []}
    results_cache = {lang: {} for lang in valid_langs}

    # =========================================================
    # Phase 1: 加载 NLLB-200 (一次加载，循环多语)
    # =========================================================
    print(f"\n[{device}] Loading NLLB Model from {NLLB_PATH} ...")
    try:
        n_model = AutoModelForSeq2SeqLM.from_pretrained(
            NLLB_PATH, 
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
        ).to(device).eval()
    except:
        print("⚠️ Flash Attention 2 加载失败或不支持，回退到默认模式。")
        n_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_PATH, torch_dtype=torch.float16).to(device).eval()
    
    n_tok = AutoTokenizer.from_pretrained(NLLB_PATH)

    # 循环翻译所有语言
    for lang in valid_langs:
        target_code = SUPPORTED_LANGS[lang]
        # NLLB 支持所有列表中的语言
        n_short, n_long = translate_dataset(n_model, n_tok, data, target_code, device)
        results_cache[lang]['short'] = n_short
        results_cache[lang]['long'] = n_long

    # 卸载模型 (可选，如果后面没有其他重型操作)
    del n_model, n_tok
    torch.cuda.empty_cache()
    gc.collect()

    # =========================================================
    # Phase 2: 合并结果并保存
    # =========================================================
    print(f"\n🔄 正在合并多语言数据...")
    for idx, item in enumerate(data):
        if 'translations' not in item:
            item['translations'] = {}
        
        # 注入所有语言的翻译结果 (仅 NLLB)
        for lang in valid_langs:
            item['translations'][lang] = {
                "short_nllb": results_cache[lang]['short'][idx],
                "long_nllb":  results_cache[lang]['long'][idx]
            }

    print(f"💾 保存最终结果至: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("🎉 NLLB 翻译任务全部完成！")

if __name__ == "__main__":
    main()