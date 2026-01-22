import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict

# ==========================================
# 配置
# ==========================================
INPUT_FILE = "dataset_optimal_filtered.json"
TARGET_MODELS = ["nllb", "seamless", "qwen", "madlad"]

def calculate_4model_average(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        return

    print(f"📖 正在读取: {file_path} ...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 数据结构: stats[Language][Metric] = list of all scores from all 4 models
    # 也就是把4个模型的分数混在一起算平均
    stats = defaultdict(lambda: {'QE': [], 'CLIP': [], 'BS': []})
    
    # 全局统计 (所有语言 + 所有模型)
    global_stats = {'QE': [], 'CLIP': [], 'BS': []}

    print("⚙️ 正在聚合 4 个模型的分数...")
    
    for item in data:
        if 'translations' not in item: continue
        
        for lang, content in item['translations'].items():
            lang_key = lang.lower()
            
            for key, value in content.items():
                # 1. 基础过滤
                if not key.startswith('score_') or value is None: continue
                if not isinstance(value, (int, float)): continue
                
                # 2. 必须属于指定的 4 个模型之一
                # 检查 key 中是否包含 'nllb', 'seamless', 'qwen', 'madlad'
                if not any(m in key for m in TARGET_MODELS):
                    continue

                # 3. 识别指标并收集
                if 'comet' in key:
                    stats[lang_key]['QE'].append(value)
                    global_stats['QE'].append(value)
                elif 'visual' in key:
                    stats[lang_key]['CLIP'].append(value)
                    global_stats['CLIP'].append(value)
                elif 'bert' in key:
                    stats[lang_key]['BS'].append(value)
                    global_stats['BS'].append(value)

    # === 生成表格 ===
    rows = []
    
    # 自定义语言顺序 (可选)
    custom_order = ["uyghur", "kazakh", "kyrgyz", "tajik", "uzbek", "urdu"]
    
    # 确保只处理存在的语言
    sorted_langs = [l for l in custom_order if l in stats]
    # 如果有其他语言未在自定义列表中，也加上
    for l in sorted(stats.keys()):
        if l not in sorted_langs: sorted_langs.append(l)

    for lang in sorted_langs:
        metrics = stats[lang]
        
        # 计算平均分 (4个模型的混合平均)
        # QE, BS 乘 100
        qe_avg = np.mean(metrics['QE']) * 100 if metrics['QE'] else 0.0
        bs_avg = np.mean(metrics['BS']) * 100 if metrics['BS'] else 0.0
        clip_avg = np.mean(metrics['CLIP']) if metrics['CLIP'] else 0.0
        
        rows.append({
            "Target Language": lang.capitalize(),
            "Avg-QE": qe_avg,
            "Avg-CLIP": clip_avg,
            "Avg-BS": bs_avg
        })

    # 添加最后一行：Total Average
    rows.append({
        "Target Language": "AVERAGE",
        "Avg-QE": np.mean(global_stats['QE']) * 100 if global_stats['QE'] else 0.0,
        "Avg-CLIP": np.mean(global_stats['CLIP']) if global_stats['CLIP'] else 0.0,
        "Avg-BS": np.mean(global_stats['BS']) * 100 if global_stats['BS'] else 0.0
    })

    # 输出
    df = pd.DataFrame(rows)
    
    # 格式化
    pd.options.display.float_format = '{:.4f}'.format
    
    print("\n" + "="*60)
    print("📊 4 Model Ensemble Average (per Language)")
    print("="*60)
    print(df.to_string(index=False))
    
    # 保存
    df.to_csv("4models_average_stats.csv", index=False)
    print("\n✅ 结果已保存至 4models_average_stats.csv")

if __name__ == '__main__':
    calculate_4model_average(INPUT_FILE)