import argparse
import json
import os
import sys
import numpy as np

def load_data(file_path):
    """
    高效加载数据，兼容 JSONL 和 JSON Array
    """
    print(f"正在加载数据: {file_path} ...", file=sys.stderr)
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                # JSON Array
                data = json.load(f)
            else:
                # JSONL
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
    except Exception as e:
        print(f"数据加载失败: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"原始数据量: {len(data)} 条", file=sys.stderr)
    return data

def calculate_stats(data):
    """
    计算统计指标
    """
    # 提取关键字段
    short_scores = [x.get('short_score', 0.0) for x in data]
    long_scores = [x.get('long_score', 0.0) for x in data]
    
    # 简单按空格分词计算长度
    short_lens = [len(str(x.get('short_caption_best', '')).split()) for x in data]
    long_lens = [len(str(x.get('long_caption_best', '')).split()) for x in data]

    return {
        "short_score": np.array(short_scores),
        "long_score": np.array(long_scores),
        "short_len": np.array(short_lens),
        "long_len": np.array(long_lens)
    }

def print_report(stats):
    """
    打印纯文本统计报告 (类似 pandas describe)
    """
    print("\n" + "="*50)
    print("DATASET ANALYSIS REPORT (PURE TEXT)")
    print("="*50)
    
    metrics = ["short_score", "long_score", "short_len", "long_len"]
    percentiles = [10, 25, 50, 75, 90]
    
    # 表头
    headers = f"{'Metric':<12} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8} | " + " | ".join([f"{p}%" for p in percentiles])
    print(headers)
    print("-" * len(headers))
    
    for m in metrics:
        arr = stats[m]
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        
        # 计算分位数
        per_vals = np.percentile(arr, percentiles)
        per_str = " | ".join([f"{v:<8.4f}" if 'score' in m else f"{int(v):<8}" for v in per_vals])
        
        # 格式化输出
        if 'len' in m:
            print(f"{m:<12} | {mean_val:<8.2f} | {std_val:<8.2f} | {int(min_val):<8} | {int(max_val):<8} | {per_str}")
        else:
            print(f"{m:<12} | {mean_val:<8.4f} | {std_val:<8.4f} | {min_val:<8.4f} | {max_val:<8.4f} | {per_str}")
            
    print("="*50 + "\n")

def filter_data(data, min_score, min_short_len, min_long_len):
    """
    执行数据筛选
    """
    print(f"正在根据阈值筛选数据...", file=sys.stderr)
    print(f"  - Min Score (SigLIP): > {min_score}", file=sys.stderr)
    print(f"  - Min Length (Short): >= {min_short_len}", file=sys.stderr)
    print(f"  - Min Length (Long):  >= {min_long_len}", file=sys.stderr)
    
    filtered_data = []
    
    for item in data:
        s_score = item.get('short_score', 0.0)
        l_score = item.get('long_score', 0.0)
        s_text = str(item.get('short_caption_best', ''))
        l_text = str(item.get('long_caption_best', ''))
        
        s_len = len(s_text.split())
        l_len = len(l_text.split())
        
        # 筛选逻辑：分数都要达标，长度都要达标
        if (s_score > min_score and l_score > min_score and 
            s_len >= min_short_len and l_len >= min_long_len):
            
            # 为了节省空间，可以移除 'short_candidates' 和 'long_candidates' 字段
            # 如果你想保留它们，注释掉下面两行
            if 'short_candidates' in item: del item['short_candidates']
            if 'long_candidates' in item: del item['long_candidates']
            
            filtered_data.append(item)
            
    # 统计
    total = len(data)
    kept = len(filtered_data)
    dropped = total - kept
    rate = (dropped / total) * 100 if total > 0 else 0
    
    print(f"\n筛选结果:")
    print(f"  - 原始数量: {total}")
    print(f"  - 保留数量: {kept}")
    print(f"  - 剔除数量: {dropped} ({rate:.2f}%)")
    
    return filtered_data

def main():
    parser = argparse.ArgumentParser(description="Analyze and Filter Dataset (No GUI)")
    parser.add_argument("--input_file", type=str, required=True, help="Stage 2 输出文件")
    parser.add_argument("--output_file", type=str, required=True, help="清洗后的最终文件")
    
    # 默认阈值 (如果不指定，脚本只做分析，不做强力清洗，min_score=0.0)
    # 建议先跑一次看报告，第二次跑带上具体的 --min_score
    parser.add_argument("--min_score", type=float, default=0.0, help="SigLIP 分数阈值 (建议参考 25% 分位数)")
    parser.add_argument("--min_short_len", type=int, default=5, help="最短短描述词数")
    parser.add_argument("--min_long_len", type=int, default=15, help="最短长描述词数")
    
    args = parser.parse_args()

    # 1. 加载
    data = load_data(args.input_file)
    if not data:
        print("数据为空，退出。")
        return

    # 2. 统计分析
    stats = calculate_stats(data)
    print_report(stats)
    
    # 3. 筛选
    # 如果用户没有指定 --min_score (默认为0)，脚本实际上只过滤长度极短的异常值
    clean_data = filter_data(data, args.min_score, args.min_short_len, args.min_long_len)
    
    # 4. 保存
    print(f"\n正在保存清洗后的数据至: {args.output_file} ...", file=sys.stderr)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
    
    print("✅ 完成！", file=sys.stderr)

if __name__ == "__main__":
    main()
