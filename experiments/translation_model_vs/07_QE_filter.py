import json
import sys
import argparse
from collections import defaultdict

# ==========================================
# 🔥 [严选阈值配置] 🔥
# 只有超过这些分数的翻译才有资格进入"决赛"
# ==========================================
# ==========================================
# 🔥 [黄金阈值配置] 🔥
# 这些值是根据学术经验设定的"高质量"基准线
# ==========================================

# 1. 语义一致性 (BERTScore): 最重要的指标
# 如果回译都对不上，说明翻译完全错了。
THRESHOLD_BERT = 0.88

# 2. 翻译质量 (COMET-Kiwi): 
# 保证译文流畅、语法正确。
THRESHOLD_COMET = 0.72

# 3. 视觉一致性 (CLIP Score): 
# 防止严重幻觉 (Hallucination)。
THRESHOLD_CLIP = 0.22

def load_data(file_path):
    print(f"📖 读取数据: {file_path} ...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if f.read(1) == '[':
                f.seek(0); return json.load(f)
            f.seek(0); return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"❌ 读取失败: {e}"); sys.exit(1)

def select_best_candidate(trans_obj, type_prefix):
    """
    输入: trans_obj (某个语言的所有翻译字段), type_prefix ('short' or 'long')
    输出: (best_text, best_model_name, best_scores) or (None, None, None)
    """
    # 1. 定义参赛选手
    candidates = []
    
    # 选手 A: NLLB
    if f"{type_prefix}_nllb" in trans_obj:
        candidates.append({
            "model": "nllb",
            "text": trans_obj.get(f"{type_prefix}_nllb"),
            "bert": trans_obj.get(f"score_bert_{type_prefix}_nllb", -1),
            "comet": trans_obj.get(f"score_comet_{type_prefix}_nllb", -1),
            "visual": trans_obj.get(f"score_visual_{type_prefix}_nllb", -1)
        })
        
    # 选手 B: Seamless (如果存在)
    if f"{type_prefix}_seamless" in trans_obj:
        candidates.append({
            "model": "seamless",
            "text": trans_obj.get(f"{type_prefix}_seamless"),
            "bert": trans_obj.get(f"score_bert_{type_prefix}_seamless", -1),
            "comet": trans_obj.get(f"score_comet_{type_prefix}_seamless", -1),
            "visual": trans_obj.get(f"score_visual_{type_prefix}_seamless", -1)
        })

    # 2. 资格赛 (过滤掉不及格的)
    qualified = []
    for cand in candidates:
        if not cand['text']: continue
        # 必须同时满足三个硬指标
        if (cand['bert'] >= THRESHOLD_BERT and 
            cand['comet'] >= THRESHOLD_COMET and 
            cand['visual'] >= THRESHOLD_CLIP):
            qualified.append(cand)

    if not qualified:
        return None, None, None

    # 3. 决赛 (COMET 决胜负)
    # 按 COMET 分数从高到低排序，取第一个
    best_cand = sorted(qualified, key=lambda x: x['comet'], reverse=True)[0]
    
    return best_cand['text'], best_cand['model'], {
        "bert": best_cand['bert'],
        "comet": best_cand['comet'],
        "visual": best_cand['visual']
    }

def process_dataset(data, output_file):
    print("🚀 开始执行 [优中选优] 策略...")
    
    final_data = []
    stats = {
        "total": len(data),
        "kept": 0,
        "lang_stats": defaultdict(lambda: {"short_nllb":0, "short_seamless":0, "long_nllb":0, "long_seamless":0})
    }

    for item in data:
        if 'translations' not in item: continue
        
        new_translations = {}
        has_content = False
        
        # 遍历每种语言
        for lang, trans_obj in item['translations'].items():
            lang_result = {}
            
            # --- 处理 Short Caption ---
            s_text, s_model, s_scores = select_best_candidate(trans_obj, "short")
            if s_text:
                lang_result["short_translation"] = s_text
                lang_result["short_model"] = s_model # 记录是谁赢了
                lang_result["short_scores"] = s_scores
                stats["lang_stats"][lang][f"short_{s_model}"] += 1
            
            # --- 处理 Long Caption ---
            l_text, l_model, l_scores = select_best_candidate(trans_obj, "long")
            if l_text:
                lang_result["long_translation"] = l_text
                lang_result["long_model"] = l_model
                lang_result["long_scores"] = l_scores
                stats["lang_stats"][lang][f"long_{l_model}"] += 1
            
            # 只有当该语言至少保留了一个 caption 时，才写入
            if lang_result:
                new_translations[lang] = lang_result
                has_content = True

        if has_content:
            # 构建极其干净的最终数据结构
            final_item = {
                "image_id": item['image_id'],
                "path": item['path'],
                # 源英文
                "src_short": item.get('short_caption_best'),
                "src_long": item.get('long_caption_best'),
                # 筛选后的多语种翻译
                "translations": new_translations
            }
            final_data.append(final_item)
            stats["kept"] += 1

    # 保存
    print(f"\n💾 保存最终黄金数据集至: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    # 打印战报
    print("\n" + "="*50)
    print("🏆 FINAL DATASET STATISTICS")
    print("="*50)
    print(f"原始数据 : {stats['total']}")
    print(f"最终保留 : {stats['kept']} (保留率: {stats['kept']/stats['total']*100:.2f}%)")
    print("-" * 50)
    print("各语言模型胜出分布 (Winning Model Distribution):")
    print(f"{'Language':<12} | {'Short NLLB':<10} {'Seamless':<10} | {'Long NLLB':<10} {'Seamless':<10}")
    print("-" * 60)
    
    for lang, counts in stats["lang_stats"].items():
        s_n = counts['short_nllb']
        s_s = counts['short_seamless']
        l_n = counts['long_nllb']
        l_s = counts['long_seamless']
        print(f"{lang:<12} | {s_n:<10} {s_s:<10} | {l_n:<10} {l_s:<10}")
        
    print("="*50)
    print("✨ 数据集构建完成。这是绝对的最优解。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 输入必须是上一步带有分数的 json
    parser.add_argument("--input_file", type=str, default="dataset_scored_final.json")
    parser.add_argument("--output_file", type=str, default="dataset_golden.json")
    args = parser.parse_args()

    data = load_data(args.input_file)
    process_dataset(data, args.output_file)
    
    
    
"""
python 06_filter_final.py \
  --input_file "dataset_scored_final.json" \
  --output_file "dataset_optimal_filtered.json"
"""


"""

核心算法流程：
1、资格赛（Hard Filtering）：
    首先检查 NLLB 和 Seamless 的翻译是否都达到了**“及格线”**（即上一轮设定的 BERT>0.88, CLIP>0.22, COMET>0.72）。
    如果某一个模型没及格，直接淘汰。
    如果两个都没及格，这条数据对应的翻译任务（Short 或 Long）直接废弃。
2、决赛（Winner Selection）：
    如果两个模型都及格了，谁更好？
    判决标准：比较 COMET-Kiwi 分数。
    理由：在都保证了语义（BERT）和视觉（CLIP）正确的前提下，COMET-Kiwi 分数越高，代表译文越地道、越符合人类阅读习惯。我们选取 COMET 分数更高的那个作为最终结果。
3、维吾尔语特例：
    由于只有 NLLB，它直接进入“资格赛”。及格就留，不及格就扔。
"""