import json
import os
import numpy as np
from collections import defaultdict

# ==========================================
# 配置部分
# ==========================================
# 请确保文件名与您实际的数据文件名一致
TARGET_FILE = "translated_data.json"

def calculate_metrics(file_path):
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        return

    print(f"📖 正在读取数据: {file_path} ...")
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            # 兼容处理可能存在的 BOM 头或格式问题
            f.seek(0)
            char = f.read(1)
            f.seek(0)
            if char == '[':
                data = json.load(f)
            else:
                data = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            print(f"❌ JSON 读取失败: {e}")
            return

    # 2. 初始化统计容器
    # stats[语言][指标] = [分数列表]
    stats = defaultdict(lambda: defaultdict(list))
    count = 0

    print("⚙️ 正在计算平均分...")
    
    for item in data:
        if 'translations' not in item:
            continue
        
        count += 1
        for lang, content in item['translations'].items():
            # 遍历该语言下的所有键值对 (例如 score_comet_short_nllb, score_visual_long_seamless 等)
            for key, value in content.items():
                # 过滤无效数据：必须是 score_ 开头，且值不为 None
                if not key.startswith('score_') or value is None:
                    continue
                
                # 确保是数值类型
                if not isinstance(value, (int, float)):
                    continue

                # === 核心映射逻辑 ===
                # 根据键名中的关键字归类到三个指标
                if 'comet' in key:
                    stats[lang]['comet'].append(value)
                elif 'visual' in key:  # 对应 CLIP Visual Score
                    stats[lang]['clip'].append(value)
                elif 'bert' in key:    # 对应 BERTScore
                    stats[lang]['bert'].append(value)

    # 3. 打印表格
    print("\n" + "="*80)
    # 表头：左边第一列是语言，右边三列是 COMET, CLIP, BERTScore
    # 注意：为了阅读方便，通常将 COMET x 100
    headers = ["Language", "COMET (x100)", "CLIP (Visual)", "BERTScore"]
    print(f"{headers[0]:<15} | {headers[1]:<15} | {headers[2]:<15} | {headers[3]:<15}")
    print("-" * 80)

    # 用于计算底部的总平均 (Global Average)
    global_scores = {'comet': [], 'clip': [], 'bert': []}

    # 按语言字母顺序排序输出
    for lang in sorted(stats.keys()):
        metrics = stats[lang]
        
        # 计算该语言的平均分 (如果列表为空则为0)
        # COMET 乘 100 以符合常见展示习惯 (e.g. 74.5)
        avg_comet = np.mean(metrics['comet']) * 100 if metrics['comet'] else 0.0
        avg_clip = np.mean(metrics['clip']) if metrics['clip'] else 0.0
        avg_bert = np.mean(metrics['bert']) if metrics['bert'] else 0.0
        
        # 打印当前语言行
        print(f"{lang:<15} | {avg_comet:<15.4f} | {avg_clip:<15.4f} | {avg_bert:<15.4f}")

        # 收集数据到全局列表
        global_scores['comet'].extend(metrics['comet'])
        global_scores['clip'].extend(metrics['clip'])
        global_scores['bert'].extend(metrics['bert'])

    print("-" * 80)

    # 4. 计算并打印最后一行：所有平均分 (AVERAGE)
    all_avg_comet = np.mean(global_scores['comet']) * 100 if global_scores['comet'] else 0.0
    all_avg_clip = np.mean(global_scores['clip']) if global_scores['clip'] else 0.0
    all_avg_bert = np.mean(global_scores['bert']) if global_scores['bert'] else 0.0

    print(f"{'AVERAGE':<15} | {all_avg_comet:<15.4f} | {all_avg_clip:<15.4f} | {all_avg_bert:<15.4f}")
    print("="*80)
    print(f"✅ 统计完成，共处理 {count} 条图片数据。")

if __name__ == '__main__':
    calculate_metrics(TARGET_FILE)