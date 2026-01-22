import json
import os
import pandas as pd
import numpy as np

# =================配置区域=================
# 指定要分析的文件名列表
TARGET_FILES = [
    "internvl3-8b_v2.jsonl",
    "llava-1.5-7b_v2.jsonl",
    "llava-1.6-7b_v2.jsonl",
    "qwen2.5-vl-7b_v2.jsonl",
    "qwen3-vl-8b_v2.jsonl"
]

# 如果文件不在当前目录，请修改这里
DATA_DIR = "./" 
# =========================================

def calculate_top3_avg(candidates, score_key):
    """
    通用函数：从候选列表中提取指定分数(score_key)最高的3个，计算平均值
    """
    if not candidates or not isinstance(candidates, list):
        return 0.0
    
    scores = []
    for cand in candidates:
        # 尝试获取指定key的分数，如果没有则默认为0
        # 兼容逻辑：如果找 siglip_score 但只有 score，则取 score
        val = 0.0
        if score_key == 'siglip_score':
            val = cand.get('siglip_score', cand.get('score', 0.0))
        else:
            val = cand.get(score_key, 0.0)
            
        scores.append(val)
    
    # 降序排列
    scores.sort(reverse=True)
    # 取前3个
    top_n = scores[:3]
    
    if not top_n:
        return 0.0
        
    return np.mean(top_n)

def calculate_length_stats(lengths, prefix):
    """
    计算长度统计指标
    """
    if not lengths:
        return {
            f'{prefix} Len: Avg': 0, f'{prefix} Len: Med': 0, 
            f'{prefix} Len: Min': 0, f'{prefix} Len: Max': 0,
            f'{prefix} Len: Q1': 0,  f'{prefix} Len: Q3': 0
        }
    
    return {
        f'{prefix} Len: Avg': np.mean(lengths),
        f'{prefix} Len: Min': np.min(lengths),
        f'{prefix} Len: Max': np.max(lengths),
        f'{prefix} Len: Med': np.median(lengths),
        f'{prefix} Len: Q1': np.percentile(lengths, 25),
        f'{prefix} Len: Q3': np.percentile(lengths, 75)
    }

def analyze_single_file(filepath):
    filename = os.path.basename(filepath)
    
    # 累加器初始化
    data = {
        'count': 0,
        # SigLIP (Ranking Metric)
        'short_siglip_best_sum': 0.0, 'short_siglip_top3_sum': 0.0,
        'long_siglip_best_sum': 0.0,  'long_siglip_top3_sum': 0.0,
        # CLIP (Evaluation Metric)
        'short_clip_best_sum': 0.0,   'short_clip_top3_sum': 0.0,
        'long_clip_best_sum': 0.0,    'long_clip_top3_sum': 0.0,
    }
    
    short_lengths = []
    long_lengths = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                try:
                    item = json.loads(line)
                    
                    # ==================== Short Caption 处理 ====================
                    s_cands = item.get('short_candidates', [])
                    
                    # 1. SigLIP Stats
                    # 兼容旧字段名 score 或 short_score
                    s_sig_best = item.get('short_score', item.get('score', 0.0))
                    s_sig_top3 = calculate_top3_avg(s_cands, 'siglip_score')
                    
                    # 2. CLIP Stats (新字段)
                    s_clip_best = item.get('short_clip_score', 0.0)
                    s_clip_top3 = calculate_top3_avg(s_cands, 'clip_score')
                    
                    # 3. Length Stats
                    s_text = item.get('short_caption_best', "")
                    short_lengths.append(len(s_text.split()) if isinstance(s_text, str) else 0)
                    
                    # ==================== Long Caption 处理 ====================
                    l_cands = item.get('long_candidates', [])
                    
                    # 1. SigLIP Stats
                    l_sig_best = item.get('long_score', 0.0)
                    l_sig_top3 = calculate_top3_avg(l_cands, 'siglip_score')
                    
                    # 2. CLIP Stats
                    l_clip_best = item.get('long_clip_score', 0.0)
                    l_clip_top3 = calculate_top3_avg(l_cands, 'clip_score')
                    
                    # 3. Length Stats
                    l_text = item.get('long_caption_best', "")
                    long_lengths.append(len(l_text.split()) if isinstance(l_text, str) else 0)
                    
                    # ==================== 累加 ====================
                    data['short_siglip_best_sum'] += s_sig_best
                    data['short_siglip_top3_sum'] += s_sig_top3
                    data['short_clip_best_sum'] += s_clip_best
                    data['short_clip_top3_sum'] += s_clip_top3
                    
                    data['long_siglip_best_sum'] += l_sig_best
                    data['long_siglip_top3_sum'] += l_sig_top3
                    data['long_clip_best_sum'] += l_clip_best
                    data['long_clip_top3_sum'] += l_clip_top3
                    
                    data['count'] += 1
                    
                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        print(f"⚠️ 文件未找到: {filename}")
        return None

    if data['count'] == 0:
        return None

    c = data['count']
    
    # 汇总结果字典
    result = {
        'File Name': filename,
        'Count': c,
        
        # Short Metrics
        'S-SigLIP Best': data['short_siglip_best_sum'] / c,
        'S-SigLIP Top3': data['short_siglip_top3_sum'] / c,
        'S-CLIP Best': data['short_clip_best_sum'] / c,
        'S-CLIP Top3': data['short_clip_top3_sum'] / c,
        
        # Long Metrics
        'L-SigLIP Best': data['long_siglip_best_sum'] / c,
        'L-SigLIP Top3': data['long_siglip_top3_sum'] / c,
        'L-CLIP Best': data['long_clip_best_sum'] / c,
        'L-CLIP Top3': data['long_clip_top3_sum'] / c,
    }
    
    # 合并长度统计
    result.update(calculate_length_stats(short_lengths, "Short"))
    result.update(calculate_length_stats(long_lengths, "Long"))
    
    return result

def main():
    results = []
    print(f"正在分析 {len(TARGET_FILES)} 个文件...\n")
    
    for filename in TARGET_FILES:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            res = analyze_single_file(filepath)
            if res:
                results.append(res)
        else:
            print(f"🔍 未在当前目录找到 {filename}，跳过...")

    if not results:
        print("❌ 没有生成任何有效数据。")
        return

    df = pd.DataFrame(results)
    
    # 格式化数值：长度相关保留1位小数，分数保留4位
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, float]:
            if 'Len' in col:
                df[col] = df[col].apply(lambda x: round(x, 1))
            else:
                df[col] = df[col].apply(lambda x: round(x, 4))

    # ================= 输出表格 =================
    
    # --- 1. Short Caption 分析 ---
    print("=" * 100)
    print("📊 SHORT Caption (短描述) - 综合评分与长度分析")
    print("=" * 100)
    
    cols_short = [
        'File Name', 
        'S-SigLIP Best', 'S-CLIP Best',  # 评分
        'Short Len: Avg', 'Short Len: Med', 'Short Len: Min', 'Short Len: Max', 
        'Short Len: Q1', 'Short Len: Q3'
    ]
    
    # 检查列是否存在（防止某些旧文件没有CLIP数据导致报错）
    cols_short = [c for c in cols_short if c in df.columns]
    
    df_short = df[cols_short].copy()
    # 按照 SigLIP 分数排序
    if 'S-SigLIP Best' in df_short.columns:
        df_short = df_short.sort_values(by='S-SigLIP Best', ascending=False)
    
    try:
        print(df_short.to_markdown(index=False, tablefmt="grid"))
    except ImportError:
        print(df_short.to_string(index=False))
    print("\n")
    
    # --- 2. Long Caption 分析 ---
    print("=" * 100)
    print("📊 LONG Caption (长描述) - 综合评分与长度分析")
    print("=" * 100)
    
    cols_long = [
        'File Name', 
        'L-SigLIP Best', 'L-CLIP Best', 
        'Long Len: Avg', 'Long Len: Med', 'Long Len: Min', 'Long Len: Max',
        'Long Len: Q1', 'Long Len: Q3'
    ]
    
    cols_long = [c for c in cols_long if c in df.columns]
    
    df_long = df[cols_long].copy()
    if 'L-SigLIP Best' in df_long.columns:
        df_long = df_long.sort_values(by='L-SigLIP Best', ascending=False)
    
    try:
        print(df_long.to_markdown(index=False, tablefmt="grid"))
    except ImportError:
        print(df_long.to_string(index=False))

if __name__ == "__main__":
    main()
