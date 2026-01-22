import json
import sys
import os
import argparse  # ✅ 补上了这个关键的导入

# ==========================================
# 🔥 [最优参数硬编码] 🔥
# ==========================================

# 1. 质量阈值 (SigLIP Score)
# 设定 0.70 以剔除约 10% 的尾部差数据，保留 Top 90% 精品。
MIN_SCORE = 0.90

# 2. 长度阈值 (Word Count)
# Short: 8-20 (完美适配 NLLB/Seamless 翻译舒适区)
SHORT_RANGE = (8, 20)

# Long: 25-45 (强迫长描述必须包含足够细节，且防止幻觉)
LONG_RANGE = (25, 45)

# ==========================================

def count_words(text):
    """按空格分词计算长度"""
    if not text: return 0
    return len(text.strip().split())

def filter_dataset(input_file, output_file):
    print(f"📖 正在读取原始数据: {input_file} ...")
    data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
            else:
                for line in f:
                    if line.strip(): data.append(json.loads(line))
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    total = len(data)
    print(f"📊 原始数据量: {total}")
    print("=" * 60)
    print(f"🚀 执行 [最优数据集] 过滤策略:")
    print(f"  1. [Quality] SigLIP Score  > {MIN_SCORE} (精英筛选)")
    print(f"  2. [Short]   Word Count    : {SHORT_RANGE[0]} - {SHORT_RANGE[1]} (翻译友好)")
    print(f"  3. [Long]    Word Count    : {LONG_RANGE[0]} - {LONG_RANGE[1]} (细节丰富)")
    print("=" * 60)

    filtered_data = []
    stats = {
        "score_low": 0,
        "short_len_err": 0,
        "long_len_err": 0,
        "total_dropped": 0
    }

    for item in data:
        s_text = item.get('short_caption_best', '')
        l_text = item.get('long_caption_best', '')
        s_score = item.get('short_score', 0.0)
        l_score = item.get('long_score', 0.0)
        
        s_len = count_words(s_text)
        l_len = count_words(l_text)
        
        is_valid = True
        
        # 1. 严格的分数过滤 (Short 和 Long 必须同时达标)
        if s_score <= MIN_SCORE or l_score <= MIN_SCORE:
            stats["score_low"] += 1
            is_valid = False
            
        # 2. Short 长度过滤
        if not (SHORT_RANGE[0] <= s_len <= SHORT_RANGE[1]):
            stats["short_len_err"] += 1
            is_valid = False
            
        # 3. Long 长度过滤
        if not (LONG_RANGE[0] <= l_len <= LONG_RANGE[1]):
            stats["long_len_err"] += 1
            is_valid = False
            
        if is_valid:
            # 清洗数据，移除冗余字段
            clean_item = {
                "image_id": item.get('image_id'),
                "path": item.get('path'),
                "wnid": item.get('wnid', ''),
                "label_name": item.get('label_name', ''),
                "width": item.get('width'),
                "height": item.get('height'),
                "short_caption_best": s_text,
                "short_score": s_score,
                "long_caption_best": l_text,
                "long_score": l_score,
            }
            filtered_data.append(clean_item)
        else:
            stats["total_dropped"] += 1

    # 保存
    print(f"💾 正在保存清洗后的数据至: {output_file} ...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    # 最终报告
    kept = len(filtered_data)
    rate = (kept / total) * 100 if total > 0 else 0

    print("\n" + "="*60)
    print("✅ FILTERING REPORT (FINAL)")
    print("="*60)
    print(f"原始数量 : {total}")
    print(f"保留数量 : {kept}")
    print(f"保留率   : {rate:.2f}%")
    print(f"剔除总数 : {stats['total_dropped']}")
    print("-" * 60)
    print(f"剔除原因分析 (存在重叠):")
    print(f"  - 分数过低 (<{MIN_SCORE})        : {stats['score_low']}")
    print(f"  - Short 长度不符 {SHORT_RANGE} : {stats['short_len_err']}")
    print(f"  - Long  长度不符 {LONG_RANGE} : {stats['long_len_err']}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    filter_dataset(args.input_file, args.output_file)
