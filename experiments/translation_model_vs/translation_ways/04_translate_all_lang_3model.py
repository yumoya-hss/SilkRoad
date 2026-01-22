import json
import sys
import os
import torch
import argparse
import gc
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM,
    SeamlessM4Tv2ForTextToText
)

# ==========================================
# 0. 用户硬编码配置区域 (直接修改这里)
# ==========================================

# 1. SeamlessM4T v2 (Large)
SEAMLESS_PATH = "models/seamless-m4t-v2-large"

# 2. NLLB-200 (3.3B)
NLLB_PATH = "models/nllb-200-3.3B"

# 3. Qwen2.5-32B (Instruct-AWQ) 
# 建议使用 AWQ 量化版以节省显存
QWEN_PATH = "outputs/Qwen3-32B"

# 批处理大小
BATCH_SIZE_SEQ2SEQ = 64  # NLLB, Seamless
BATCH_SIZE_LLM = 16      # Qwen (显存占用较大，建议调小)

# ==========================================
# 1. 语言代码映射表
# ==========================================
SUPPORTED_LANGS = {
    # --- 中亚 ---
    "uzbek": "uzn_Latn", "kazakh": "kaz_Cyrl", "kyrgyz": "kir_Cyrl", "tajik": "tgk_Cyrl",
    # --- 南亚 ---
    "urdu": "urd_Arab", "bengali": "ben_Beng", "pashto": "pbt_Arab", "hindi": "hin_Deva",
    "nepali": "npi_Deva", "marathi": "mar_Deva", "telugu": "tel_Telu", "tamil": "tam_Taml",
    # --- 东南亚 ---
    "vietnamese": "vie_Latn", "thai": "tha_Thai", "indonesian": "ind_Latn", "khmer": "khm_Khmr",
    "lao": "lao_Laoo", "burmese": "mya_Mymr", "malay": "zsm_Latn",
    # --- 中东/西亚 ---
    "persian": "pes_Arab", "arabic": "arb_Arab", "turkish": "tur_Latn", "hebrew": "heb_Hebr",
    # --- 非洲 ---
    "swahili": "swh_Latn", "yoruba": "yor_Latn", "zulu": "zul_Latn", "amharic": "amh_Ethi", "hausa": "hau_Latn",
    # --- 东亚/其它 ---
    "uyghur": "uig_Arab", "mongolian": "mon_Cyrl", "korean": "kor_Hang", "japanese": "jpn_Jpan", "chinese": "zho_Hans",
}

# Seamless 映射
NLLB_TO_SEAMLESS_MAP = {
    "uzn_Latn": "uzn", "kaz_Cyrl": "kaz", "kir_Cyrl": "kir", "tgk_Cyrl": "tgk",
    "urd_Arab": "urd", "ben_Beng": "ben", "pbt_Arab": "pbt", "hin_Deva": "hin",
    "npi_Deva": "npi", "mar_Deva": "mar", "tel_Telu": "tel", "tam_Taml": "tam",
    "vie_Latn": "vie", "tha_Thai": "tha", "ind_Latn": "ind", "khm_Khmr": "khm",
    "lao_Laoo": "lao", "mya_Mymr": "mya", 
    "pes_Arab": "pes", "arb_Arab": "arb", "tur_Latn": "tur", "heb_Hebr": "heb",
    "swh_Latn": "swh", "yor_Latn": "yor", "zul_Latn": "zul", "amh_Ethi": "amh",
    "mon_Cyrl": "mon", "kor_Hang": "kor", "jpn_Jpan": "jpn",
    "zho_Hans": "cmn", "zsm_Latn": "zlm", "hau_Latn": "hau", "uig_Arab": "uig",
}
SEAMLESS_SUPPORTED_TGTS = {
    "afr","amh","arb","ary","arz","asm","azj","bel","ben","bos","bul","cat","ceb","ces",
    "ckb","cmn","cmn_Hant","cym","dan","deu","ell","eng","est","eus","fin","fra","fuv",
    "gaz","gle","glg","guj","heb","hin","hrv","hun","hye","ibo","ind","isl","ita","jav",
    "jpn","kan","kat","kaz","khk","khm","kir","kor","lao","lit","lug","luo","lvs","mai",
    "mal","mar","mkd","mlt","mni","mya","nld","nno","nob","npi","nya","ory","pan","pbt",
    "pes","pol","por","ron","rus","sat","slk","slv","sna","snd","som","spa","srp","swe",
    "swh","tam","tel","tgk","tgl","tha","tur","ukr","urd","uzn","vie","yor","yue","zlm","zul",
    "hau"
}

# 🔥 [核心优化] Qwen 语言脚本映射表
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
# 3. Seq2Seq 翻译核心 (Seamless, NLLB)
# ==========================================
def translate_seq2seq(model, tokenizer, data, model_type, lang_key, target_code, device):
    print(f"   >> [{model_type.upper()}] Target: {lang_key} ...")
    
    short_results, long_results = [], []
    
    seamless_lang = NLLB_TO_SEAMLESS_MAP.get(target_code, target_code[:3]) if model_type == 'seamless' else None
    nllb_bos_id = tokenizer.convert_tokens_to_ids(target_code) if model_type == 'nllb' else None

    for i in tqdm(range(0, len(data), BATCH_SIZE_SEQ2SEQ), desc=f"   Processing ({model_type})"):
        batch = data[i : i + BATCH_SIZE_SEQ2SEQ]
        short_src = [item.get('short_caption_best', "") for item in batch]
        long_src = [item.get('long_caption_best', "") for item in batch]
        
        def run_gen(texts, max_len):
            valid_indices = [i for i, t in enumerate(texts) if t.strip()]
            if not valid_indices: return [""] * len(texts)
            
            inputs_text = [texts[i] for i in valid_indices]
            inputs = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            gen_kwargs = {
                "max_new_tokens": max_len,
                "num_beams": 5,
                "do_sample": False
            }
            
            if model_type == 'seamless': gen_kwargs["tgt_lang"] = seamless_lang
            elif model_type == 'nllb': gen_kwargs["forced_bos_token_id"] = nllb_bos_id
            
            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kwargs)
            
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            
            final_res = [""] * len(texts)
            for idx, res in zip(valid_indices, decoded):
                final_res[idx] = res.strip()
            return final_res

        short_results.extend(run_gen(short_src, 128)) 
        long_results.extend(run_gen(long_src, 256)) 
        
    return short_results, long_results

# ==========================================
# 4. LLM 翻译核心 (Qwen2.5-32B)
# ==========================================
def translate_llm_qwen(model, tokenizer, data, lang_key, device):
    """
    Qwen2.5-32B 翻译专用函数
    """
    target_lang_desc = QWEN_LANG_SCRIPT_MAP.get(lang_key, lang_key.capitalize())
    print(f"   >> [QWEN-32B] Target: {target_lang_desc} ...")
    
    short_results, long_results = [], []

    # 构造最优 System Prompt
    sys_prompt = (
        f"You are a professional translator and native speaker of {target_lang_desc}.\n"
        f"Your task is to translate an English image description (caption) into {target_lang_desc}.\n\n"
        "### Requirements:\n"
        "1. **Accuracy**: Maintain the exact meaning of the original English text.\n"
        "2. **Fluency**: Use natural, native phrasing. Do not translate word-for-word.\n"
        "3. **Script**: Use the official standard script for the target language.\n"
        "4. **Output**: Output ONLY the translated text. No 'Here is the translation', no pinyin, no notes.\n"
    )

    for i in tqdm(range(0, len(data), BATCH_SIZE_LLM), desc="   Processing (Qwen)"):
        batch = data[i : i + BATCH_SIZE_LLM]
        short_src = [item.get('short_caption_best', "") for item in batch]
        long_src = [item.get('long_caption_best', "") for item in batch]

        def run_gen(texts, max_len):
            valid_indices = [i for i, t in enumerate(texts) if t.strip()]
            if not valid_indices: return [""] * len(texts)
            
            prompts = []
            for idx in valid_indices:
                src_text = texts[idx]
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": src_text}
                ]
                text_input = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(text_input)

            tokenizer.padding_side = "left" 
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_len,
                    temperature=0.1,  
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            final_res = [""] * len(texts)
            for idx, res in zip(valid_indices, decoded):
                clean = res.replace("Translation:", "").replace("翻译：", "").strip()
                final_res[idx] = clean
            return final_res

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
    parser.add_argument("--langs", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    # 1. 解析语言
    valid_langs = [l.strip().lower() for l in args.langs.split(',') if l.strip().lower() in SUPPORTED_LANGS]
    if not valid_langs: return

    device = f"cuda:{args.gpu_id}"
    data = load_data(args.input_file)
    
    # 结果缓存 (只保留3个模型)
    results_cache = {lang: {m: {} for m in ['seamless', 'nllb', 'qwen']} for lang in valid_langs}

    # ----------------------------------------
    # Phase 1: SeamlessM4T v2
    # ----------------------------------------
    print(f"\n[{device}] Phase 1: Loading SeamlessM4T...")
    try:
        s_model = SeamlessM4Tv2ForTextToText.from_pretrained(SEAMLESS_PATH, torch_dtype=torch.float16).to(device).eval()
        s_tok = AutoTokenizer.from_pretrained(SEAMLESS_PATH)
        
        for lang in valid_langs:
            target_code = SUPPORTED_LANGS[lang]
            seamless_code = NLLB_TO_SEAMLESS_MAP.get(target_code, target_code[:3])
            if seamless_code in SEAMLESS_SUPPORTED_TGTS:
                s, l = translate_seq2seq(s_model, s_tok, data, 'seamless', lang, target_code, device)
                results_cache[lang]['seamless']['short'], results_cache[lang]['seamless']['long'] = s, l
            else:
                print(f"⚠️ Seamless 不支持 {lang}, 填空。")
                results_cache[lang]['seamless']['short'] = [""] * len(data)
                results_cache[lang]['seamless']['long'] = [""] * len(data)
        del s_model, s_tok; torch.cuda.empty_cache(); gc.collect()
    except Exception as e: print(f"❌ Seamless 失败: {e}")

    # ----------------------------------------
    # Phase 2: NLLB-200
    # ----------------------------------------
    print(f"\n[{device}] Phase 2: Loading NLLB...")
    try:
        n_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_PATH, torch_dtype=torch.float16).to(device).eval()
        n_tok = AutoTokenizer.from_pretrained(NLLB_PATH)
        
        for lang in valid_langs:
            target_code = SUPPORTED_LANGS[lang]
            s, l = translate_seq2seq(n_model, n_tok, data, 'nllb', lang, target_code, device)
            results_cache[lang]['nllb']['short'], results_cache[lang]['nllb']['long'] = s, l
        del n_model, n_tok; torch.cuda.empty_cache(); gc.collect()
    except Exception as e: print(f"❌ NLLB 失败: {e}")

    # ----------------------------------------
    # Phase 3: Qwen2.5-32B (LLM)
    # ----------------------------------------
    print(f"\n[{device}] Phase 3: Loading Qwen2.5-32B...")
    try:
        q_model = AutoModelForCausalLM.from_pretrained(QWEN_PATH, torch_dtype=torch.float16, device_map=device, trust_remote_code=True).eval()
        q_tok = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
        if q_tok.pad_token is None: q_tok.pad_token = q_tok.eos_token
        
        for lang in valid_langs:
            s, l = translate_llm_qwen(q_model, q_tok, data, lang, device)
            results_cache[lang]['qwen']['short'], results_cache[lang]['qwen']['long'] = s, l
        del q_model, q_tok; torch.cuda.empty_cache(); gc.collect()
    except Exception as e: print(f"❌ Qwen 失败: {e}")

    # ----------------------------------------
    # 合并保存
    # ----------------------------------------
    print(f"\n🔄 正在合并 3 模型结果...")
    for idx, item in enumerate(data):
        if 'translations' not in item: item['translations'] = {}
        
        for lang in valid_langs:
            item['translations'][lang] = {
                "short_seamless": results_cache[lang]['seamless']['short'][idx],
                "long_seamless":  results_cache[lang]['seamless']['long'][idx],
                "short_nllb":     results_cache[lang]['nllb']['short'][idx],
                "long_nllb":      results_cache[lang]['nllb']['long'][idx],
                "short_qwen":     results_cache[lang]['qwen']['short'][idx],
                "long_qwen":      results_cache[lang]['qwen']['long'][idx],
            }

    print(f"💾 保存至: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("🎉 全流程完成！")

if __name__ == "__main__":
    main()