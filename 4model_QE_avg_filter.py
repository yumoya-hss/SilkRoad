import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict

# ==========================================
# 配置
# ==========================================
# 🔥 请确保这里是【筛选后】的 JSON 文件名
INPUT_FILE = "dataset_optimal_filtered.json" 

def calculate_filtered_average(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        return

    print(f"📖 正在读取筛选后的数据: {file_path} ...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 数据结构: stats[Language][Metric] = list of scores
    stats = defaultdict(lambda: {'QE': [], 'CLIP': [], 'BS': []})
    global_stats = {'QE': [], 'CLIP': [], 'BS': []}

    print("⚙️ 正在计算平均分...")
    
    for item in data:
        if 'translations' not in item: continue
        
        for lang, content in item['translations'].items():
            lang_key = lang.lower()
            
            # 检查 Short Caption 的分数
            if 'short_scores' in content:
                scores = content['short_scores']
                # 收集 QE (COMET)
                if scores.get('comet') is not None:
                    stats[lang_key]['QE'].append(scores['comet'])
                    global_stats['QE'].append(scores['comet'])
                # 收集 BS (BERTScore)
                if scores.get('bert') is not None:
                    stats[lang_key]['BS'].append(scores['bert'])
                    global_stats['BS'].append(scores['bert'])
                # 收集 CLIP (Visual)
                if scores.get('visual') is not None:
                    stats[lang_key]['CLIP'].append(scores['visual'])
                    global_stats['CLIP'].append(scores['visual'])

            # 检查 Long Caption 的分数
            if 'long_scores' in content:
                scores = content['long_scores']
                # 收集 QE
                if scores.get('comet') is not None:
                    stats[lang_key]['QE'].append(scores['comet'])
                    global_stats['QE'].append(scores['comet'])
                # 收集 BS
                if scores.get('bert') is not None:
                    stats[lang_key]['BS'].append(scores['bert'])
                    global_stats['BS'].append(scores['bert'])
                # 收集 CLIP
                if scores.get('visual') is not None:
                    stats[lang_key]['CLIP'].append(scores['visual'])
                    global_stats['CLIP'].append(scores['visual'])

    # === 生成表格 ===
    rows = []
    custom_order = ["uyghur", "kazakh", "kyrgyz", "tajik", "uzbek", "urdu"]
    
    sorted_langs = [l for l in custom_order if l in stats]
    for l in sorted(stats.keys()):
        if l not in sorted_langs: sorted_langs.append(l)

    for lang in sorted_langs:
        metrics = stats[lang]
        
        # 计算平均分 (QE/BS x 100)
        qe_avg = np.mean(metrics['QE']) * 100 if metrics['QE'] else 0.0
        bs_avg = np.mean(metrics['BS']) * 100 if metrics['BS'] else 0.0
        clip_avg = np.mean(metrics['CLIP']) if metrics['CLIP'] else 0.0
        
        rows.append({
            "Target Language": lang.capitalize(),
            "Ours-QE": qe_avg,
            "Ours-CLIP": clip_avg,
            "Ours-BS": bs_avg
        })

    # 添加 Total Average
    rows.append({
        "Target Language": "AVERAGE",
        "Ours-QE": np.mean(global_stats['QE']) * 100 if global_stats['QE'] else 0.0,
        "Ours-CLIP": np.mean(global_stats['CLIP']) if global_stats['CLIP'] else 0.0,
        "Ours-BS": np.mean(global_stats['BS']) * 100 if global_stats['BS'] else 0.0
    })

    # 输出
    df = pd.DataFrame(rows)
    pd.options.display.float_format = '{:.2f}'.format # 保留2位小数
    
    print("\n" + "="*60)
    print("📊 Ours (Filtered) Dataset Quality Analysis")
    print("="*60)
    print(df.to_string(index=False))
    
    df.to_csv("ours_filtered_quality.csv", index=False)
    print("\n✅ 结果已保存至 ours_filtered_quality.csv")

if __name__ == '__main__':
    calculate_filtered_average(INPUT_FILE)