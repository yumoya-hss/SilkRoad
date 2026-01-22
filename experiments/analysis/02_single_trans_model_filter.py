import json
import sys
import argparse
import csv
from collections import defaultdict
import numpy as np

# ==========================================
# 🔥 [严选阈值配置] (保持原始 0-1 格式用于比对) 🔥
# ==========================================
THRESHOLD_BERT = 0.90   
THRESHOLD_COMET = 0.78  
THRESHOLD_CLIP = 0.27   

# 语言代码映射
LANG_MAP = {
    'uyghur': 'ug', 'ug': 'ug',
    'kazakh': 'kk', 'kk': 'kk',
    'kirghiz': 'ky', 'kyrgyz': 'ky', 'ky': 'ky',
    'tajik': 'tg', 'tg': 'tg',
    'urdu': 'ur', 'ur': 'ur',
    'uzbek': 'uz', 'uz': 'uz'
}
TARGET_LANGS = ['ug', 'uz', 'kk', 'ky', 'tg', 'ur']

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

def check_entry_quality(lang_data, type_prefix, target_model):
    """
    检查单条语言数据中，特定类型的翻译是否符合要求
    """
    model_key = f"{type_prefix}_model"
    score_key = f"{type_prefix}_scores"
    text_key = f"{type_prefix}_translation"

    # 1. 检查是否存在该类型的模型记录
    if model_key not in lang_data or not lang_data[model_key]:
        return None
    
    # 2. 检查模型名称是否匹配
    current_model = lang_data[model_key].lower()
    if current_model != target_model.lower():
        return None

    # 3. 获取分数 (原始 0-1 分数)
    scores = lang_data.get(score_key, {})
    if not scores: return None
    
    bert = scores.get('bert', -1)
    comet = scores.get('comet', -1)
    visual = scores.get('visual', -1)

    # 4. 阈值过滤 (使用原始 0-1 阈值进行判断)
    if (bert >= THRESHOLD_BERT and 
        comet >= THRESHOLD_COMET and 
        visual >= THRESHOLD_CLIP):
        
        # 返回结果时，将分数转换为百分制 (0-100)
        return {
            "type": type_prefix,
            "text": lang_data[text_key],
            "scores": {
                "bert": bert * 100,
                "comet": comet * 100,
                "visual": visual * 100
            }
        }
    
    return None

def process_dataset(data, target_model, output_json, output_csv):
    print(f"🚀 开始筛选，目标模型: [{target_model.upper()}]")
    print(f"🎯 原始阈值设定: BERT>={THRESHOLD_BERT}, COMET>={THRESHOLD_COMET}, CLIP>={THRESHOLD_CLIP}")
    print(f"📝 输出结果将转换为百分制 (0-100)")
    
    final_data = []
    
    # 分语言统计器
    stats = defaultdict(lambda: {"bert": [], "comet": [], "visual": [], "count": 0})
    # 🔥 全局统计器 (用于计算 Total Average)
    global_stats = {"bert": [], "comet": [], "visual": [], "count": 0}
    
    csv_rows = []

    for item in data:
        if 'translations' not in item: continue
        
        new_translations = {}
        has_content = False
        
        for lang_name, lang_content in item['translations'].items():
            lang_code = LANG_MAP.get(lang_name.lower(), lang_name)
            lang_result = {}
            
            # --- 1. 检查 Short Translation ---
            res_short = check_entry_quality(lang_content, "short", target_model)
            if res_short:
                lang_result["short_translation"] = res_short['text']
                lang_result["short_model"] = target_model
                lang_result["short_scores"] = res_short['scores']
                
                s = res_short['scores']
                # 记录分语言统计
                stats[lang_code]["bert"].append(s['bert'])
                stats[lang_code]["comet"].append(s['comet'])
                stats[lang_code]["visual"].append(s['visual'])
                stats[lang_code]["count"] += 1
                
                # 🔥 记录全局统计
                global_stats["bert"].append(s['bert'])
                global_stats["comet"].append(s['comet'])
                global_stats["visual"].append(s['visual'])
                global_stats["count"] += 1
                
                csv_rows.append([
                    item.get('image_id', 'N/A'), lang_code, "Short", target_model,
                    f"{s['bert']:.2f}", f"{s['comet']:.2f}", f"{s['visual']:.2f}",
                    res_short['text']
                ])

            # --- 2. 检查 Long Translation ---
            res_long = check_entry_quality(lang_content, "long", target_model)
            if res_long:
                lang_result["long_translation"] = res_long['text']
                lang_result["long_model"] = target_model
                lang_result["long_scores"] = res_long['scores']
                
                s = res_long['scores']
                # 记录分语言统计
                stats[lang_code]["bert"].append(s['bert'])
                stats[lang_code]["comet"].append(s['comet'])
                stats[lang_code]["visual"].append(s['visual'])
                stats[lang_code]["count"] += 1

                # 🔥 记录全局统计
                global_stats["bert"].append(s['bert'])
                global_stats["comet"].append(s['comet'])
                global_stats["visual"].append(s['visual'])
                global_stats["count"] += 1
                
                csv_rows.append([
                    item.get('image_id', 'N/A'), lang_code, "Long", target_model,
                    f"{s['bert']:.2f}", f"{s['comet']:.2f}", f"{s['visual']:.2f}",
                    res_long['text']
                ])

            if lang_result:
                new_translations[lang_name] = lang_result
                has_content = True

        if has_content:
            final_item = item.copy()
            final_item['translations'] = new_translations
            final_data.append(final_item)

    # --- 保存结果 ---
    print(f"\n💾 保存过滤后的 JSON (百分制): {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"📊 保存 CSV 报表 (百分制): {output_csv}")
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image_ID", "Language", "Type", "Model", "BERT", "COMET", "CLIP", "Text"])
        writer.writerows(csv_rows)

    # --- 打印终端报表 ---
    print("\n" + "="*90)
    print(f"🏆 模型 [{target_model.upper()}] 质量分析报告 (Score: 0-100)")
    print("="*90)
    
    header = "{:<10} | {:<10} | {:<10} | {:<10} | {:<10}".format("Language", "Count", "Avg BERT", "Avg COMET", "Avg CLIP")
    print(header)
    print("-" * 90)

    existing_langs = sorted(list(stats.keys()))
    sorted_langs = [l for l in TARGET_LANGS if l in existing_langs] + [l for l in existing_langs if l not in TARGET_LANGS]

    for lang in sorted_langs:
        st = stats[lang]
        count = st['count']
        
        if count > 0:
            avg_bert = np.mean(st['bert'])
            avg_comet = np.mean(st['comet'])
            avg_clip = np.mean(st['visual'])
            print("{:<10} | {:<10} | {:<10.2f} | {:<10.2f} | {:<10.2f}".format(
                lang, count, avg_bert, avg_comet, avg_clip
            ))
        else:
            print("{:<10} | {:<10} | -          | -          | -".format(lang, 0))

    # 🔥 打印全局平均行
    print("-" * 90)
    if global_stats["count"] > 0:
        g_bert = np.mean(global_stats["bert"])
        g_comet = np.mean(global_stats["comet"])
        g_clip = np.mean(global_stats["visual"])
        print("{:<10} | {:<10} | {:<10.2f} | {:<10.2f} | {:<10.2f}".format(
            "AVERAGE", global_stats["count"], g_bert, g_comet, g_clip
        ))
    else:
        print("{:<10} | {:<10} | -          | -          | -".format("AVERAGE", 0))

    print("="*90)
    print(f"原始图片总数: {len(data)}")
    print(f"包含有效数据的图片数: {len(final_data)}")
    print(f"总计保留条目数 (Short + Long): {global_stats['count']}")
    print("="*90)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSON 文件路径")
    parser.add_argument("--model", type=str, required=True, help="要筛选的模型名称 (如: nllb, seamless)")
    parser.add_argument("--output_dir", type=str, default=".", help="输出目录")
    args = parser.parse_args()

    out_json = f"{args.output_dir}/filtered_{args.model}.json"
    out_csv = f"{args.output_dir}/report_{args.model}.csv"

    data = load_data(args.input_file)
    process_dataset(data, args.model, out_json, out_csv)
