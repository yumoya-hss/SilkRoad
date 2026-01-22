import json
import sys
import os
import torch
import argparse
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================
# 配置区域
# ==========================================
NLLB_PATH = os.environ.get("SILKROAD_NLLB_MODEL","facebook/nllb-200-3.3B")

# NLLB 语言代码映射
LANG_CODE_MAP = {
    "uyghur": "uig_Arab",
    "uzbek": "uzn_Latn",
    "kazakh": "kaz_Cyrl",
    "kyrgyz": "kir_Cyrl",
    "tajik": "tgk_Cyrl",
    "urdu": "urd_Arab",
    "chinese": "zho_Hans",
    "vietnamese": "vie_Latn",
    "mongolian": "mon_Cyrl",
    "bengali": "ben_Beng",
    "pashto": "pbt_Arab"
}

TARGET_LANG_CODE = "eng_Latn"

def load_data(file_path):
    print(f"📖 读取数据: {file_path} ...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if f.read(1) == '[':
                f.seek(0); return json.load(f)
            else:
                f.seek(0); return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"❌ 数据错误: {e}"); sys.exit(1)

def batch_translate_smart(model, tokenizer, task_items, src_lang_code, device, batch_size):
    """
    执行智能批量回译（按长度排序）
    task_items: [(data_idx, field_key, text), ...]
    Return: List of (data_idx, field_key, translated_text)
    """
    # 1. [Smart Batching] 按文本长度倒序排序，减少 Padding 浪费 
    # 倒序通常比正序稍好，因为最长的先处理，防止最后显存碎片
    sorted_tasks = sorted(task_items, key=lambda x: len(x[2]), reverse=True)
    
    results = []
    
    # 设置源语言 (这对 NLLB 至关重要)
    tokenizer.src_lang = src_lang_code
    forced_bos_id = tokenizer.convert_tokens_to_ids(TARGET_LANG_CODE)

    # 2. 批量推理
    for i in tqdm(range(0, len(sorted_tasks), batch_size), desc=f"   Processing {src_lang_code}"):
        batch = sorted_tasks[i : i + batch_size]
        batch_texts = [item[2] for item in batch] # 提取文本
        
        # 编码
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)
        
        # 生成 (回译追求语义还原，贪婪搜索 num_beams=1 最快且最忠实原文)
        with torch.inference_mode():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=128,
                num_beams=1, 
                do_sample=False
            )
        
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # 结果与原始元数据重新绑定
        for j, trans_text in enumerate(decoded):
            original_meta = batch[j] # (data_idx, field_key, original_text)
            results.append((original_meta[0], original_meta[1], trans_text.strip()))
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64) # 排序后可以尝试更大的 batch
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    
    # 1. 加载数据
    data = load_data(args.input_file)
    
    # 2. 加载模型 (FP16 + Flash Attention)
    print(f"\n🚀 加载 NLLB 模型 (FP16)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(NLLB_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            NLLB_PATH, 
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
        ).to(device).eval()
    except:
        print("⚠️ Flash Attention 不可用，使用默认加载...")
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_PATH, torch_dtype=torch.float16).to(device).eval()

    # 3. 收集任务 (按语言分组)
    lang_tasks = {} # { "uyghur": [ (idx, key, text), ... ] }
    
    print("🔍 整理任务队列...")
    count = 0
    for idx, item in enumerate(data):
        if 'translations' not in item: continue
        for lang, trans_dict in item['translations'].items():
            if lang not in LANG_CODE_MAP: continue
            
            if lang not in lang_tasks: lang_tasks[lang] = []
            
            for key, text in trans_dict.items():
                # 跳过回译(bt_)、分数(score_)和空值
                if not key.startswith('bt_') and not key.startswith('score_') and text and text.strip():
                    lang_tasks[lang].append((idx, key, text))
                    count += 1
    
    print(f"✅ 总回译任务数: {count}")

    # 4. 执行回译 (Smart Batching Pipeline)
    for lang, tasks in lang_tasks.items():
        src_code = LANG_CODE_MAP[lang]
        print(f"\n🌍 正在回译: {lang} -> English (Tasks: {len(tasks)})")
        
        # 这一步会自动排序、批量翻译
        results = batch_translate_smart(model, tokenizer, tasks, src_code, device, args.batch_size)
        
        # 回填数据 (由于我们携带了 idx 和 key，所以乱序处理也能精准回填)
        print(f"   Writing results to memory...")
        for data_idx, key, bt_text in results:
            bt_key = f"bt_{key}"
            data[data_idx]['translations'][lang][bt_key] = bt_text
            
        # 显存整理 (每个语言跑完清理一次，保持状态最佳)
        torch.cuda.empty_cache()

    # 5. 保存
    print(f"\n💾 保存最终结果至: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print("🎉 回译全部完成！")

if __name__ == "__main__":
    main()