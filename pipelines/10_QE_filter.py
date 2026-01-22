import os
import json
import sys
import argparse
from collections import defaultdict

# ==========================================
# 🔥 [严选阈值配置] 🔥
# ==========================================
THRESHOLD_BERT = 0.90   
THRESHOLD_COMET = 0.78  
THRESHOLD_CLIP = 0.27   

# 定义所有参赛模型 (用于非维吾尔语)
MODELS = ["nllb", "seamless", "qwen", "madlad"]

def load_data(file_path):
    print(f"📖 读取数据: {file_path} ...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                return json.load(f)
            else:
                return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"❌ 读取失败: {e}"); sys.exit(1)

def select_best_candidate(trans_obj, type_prefix, lang_code):
    """
    输入: trans_obj, type_prefix, lang_code
    输出: (best_text, best_model_name, best_scores) or (None, None, None)
    """
    candidates = []
    
    # 🔥 核心修改：针对维吾尔语的特殊逻辑
    # 如果是维吾尔语 (ug / uyghur)，只考虑 NLLB
    if lang_code in ['ug', 'uyghur']:
        target_models = ['nllb']
    else:
        # 其他语言，四个模型一起竞争
        target_models = MODELS

    # 1. 遍历目标模型，收集参赛选手
    for model in target_models:
        text_key = f"{type_prefix}_{model}"
        
        # 检查该模型是否有翻译文本
        if text_key in trans_obj and trans_obj[text_key]:
            # 获取分数 (如果缺失则给 -1)
            bert = trans_obj.get(f"score_bert_{type_prefix}_{model}", -1)
            comet = trans_obj.get(f"score_comet_{type_prefix}_{model}", -1)
            visual = trans_obj.get(f"score_visual_{type_prefix}_{model}", -1)
            
            candidates.append({
                "model": model,
                "text": trans_obj[text_key],
                "bert": bert,
                "comet": comet,
                "visual": visual
            })

    # 2. 资格赛 (Hard Filtering)
    qualified = []
    for cand in candidates:
        # 必须同时满足三个硬指标
        if (cand['bert'] >= THRESHOLD_BERT and 
            cand['comet'] >= THRESHOLD_COMET and 
            cand['visual'] >= THRESHOLD_CLIP):
            qualified.append(cand)

    if not qualified:
        return None, None, None

    # 3. 决赛 (Winner Takes All)
    # 按 COMET 分数从高到低排序，取第一名
    # 对于维吾尔语，因为只有 NLLB 一个候选，所以只要 qualified 列表不为空，取出来的就是 NLLB
    best_cand = sorted(qualified, key=lambda x: x['comet'], reverse=True)[0]
    
    return best_cand['text'], best_cand['model'], {
        "bert": best_cand['bert'],
        "comet": best_cand['comet'],
        "visual": best_cand['visual']
    }

def process_dataset(data, output_file):
    print("🚀 开始执行筛选策略...")
    print(f"   - 维吾尔语 (Uyghur): 仅 NLLB 独家通道")
    print(f"   - 其他语言: {MODELS} 四模型竞技")
    
    final_data = []
    stats = {
        "total": len(data),
        "kept": 0,
        "lang_stats": defaultdict(lambda: defaultdict(int))
    }

    for item in data:
        if 'translations' not in item: continue
        
        new_translations = {}
        has_content = False
        
        # 遍历每种语言
        for lang, trans_obj in item['translations'].items():
            lang_key = lang.lower()
            lang_result = {}
            
            # 传入 lang_key 以便区分策略
            
            # --- 处理 Short Caption ---
            s_text, s_model, s_scores = select_best_candidate(trans_obj, "short", lang_key)
            if s_text:
                lang_result["short_translation"] = s_text
                lang_result["short_model"] = s_model
                lang_result["short_scores"] = s_scores
                stats["lang_stats"][lang_key][f"short_{s_model}"] += 1
            
            # --- 处理 Long Caption ---
            l_text, l_model, l_scores = select_best_candidate(trans_obj, "long", lang_key)
            if l_text:
                lang_result["long_translation"] = l_text
                lang_result["long_model"] = l_model
                lang_result["long_scores"] = l_scores
                stats["lang_stats"][lang_key][f"long_{l_model}"] += 1
            
            if lang_result:
                new_translations[lang] = lang_result
                has_content = True

        if has_content:
            final_item = {
                "image_id": item['image_id'],
                "path": item['path'],
                "src_short": item.get('short_caption_best'),
                "src_long": item.get('long_caption_best'),
                "translations": new_translations
            }
            final_data.append(final_item)
            stats["kept"] += 1

    # 保存
    print(f"\n💾 保存最终黄金数据集至: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    # 打印战报
    print("\n" + "="*80)
    print("🏆 FINAL DATASET STATISTICS")
    print("="*80)
    print(f"原始数据 : {stats['total']}")
    print(f"最终保留 : {stats['kept']} (保留率: {stats['kept']/stats['total']*100:.2f}%)")
    print("-" * 80)
    
    # 动态生成表头
    headers = ["Language"]
    for m in MODELS:
        headers.append(f"S-{m[:2].upper()}") 
        headers.append(f"L-{m[:2].upper()}")
    
    header_str = "{:<12} | " + " ".join([f"{{:<6}}" for _ in range(len(headers)-1)])
    print(header_str.format(*headers))
    print("-" * 80)
    
    for lang in sorted(stats["lang_stats"].keys()):
        counts = stats["lang_stats"][lang]
        row_vals = [lang]
        for m in MODELS:
            row_vals.append(counts[f"short_{m}"])
            row_vals.append(counts[f"long_{m}"])
        
        print(header_str.format(*row_vals))
        
    print("="*80)
    print("✨ 筛选完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=os.environ.get("SILKROAD_SCORED","outputs/scored/scored.json"))
    parser.add_argument("--output_file", type=str, default=os.environ.get("SILKROAD_GOLDEN","outputs/final/golden.json"))
    args = parser.parse_args()

    data = load_data(args.input_file)
    process_dataset(data, args.output_file)