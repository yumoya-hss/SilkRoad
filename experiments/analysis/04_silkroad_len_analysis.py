import json
import os
import argparse
from collections import defaultdict
from tqdm import tqdm

# ==========================================
# 🛠️ 默认配置
# ==========================================
DEFAULT_INPUT_FILE = "dataset_optimal_filtered_v3.json"
DEFAULT_OUTPUT_DIR = "final_datasets_split"

def load_data(file_path):
    print(f"📖 正在读取数据集: {file_path} ...")
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        exit(1)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 兼容 JSON 数组或 JSONL
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                return json.load(f)
            else:
                return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        exit(1)

def split_by_language_and_length(data, output_dir):
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 已创建输出目录: {output_dir}")

    # 2. 初始化缓存容器
    # 结构: buffers[lang]['short'] = List[Dict]
    lang_buffers = defaultdict(lambda: {'short': [], 'long': []})
    
    # 3. 初始化长度统计 (用于计算平均长度)
    # 结构: len_stats[lang]['short'] = total_word_count
    len_stats = defaultdict(lambda: {'short': 0, 'long': 0})
    
    print("🚀 正在执行 [语言 + 长短句] 二级拆分及统计...")
    
    # 4. 遍历数据
    for item in tqdm(data):
        if 'translations' not in item: continue
        
        # 基础元数据
        base_meta = {
            "image_id": item.get("image_id"),
            "path": item.get("path"),
        }

        # 遍历该条目下的所有语言
        for lang, trans_content in item['translations'].items():
            if not trans_content: continue

            # --- 处理 Short Caption ---
            tgt_short = trans_content.get("short_translation")
            if tgt_short:
                short_entry = base_meta.copy()
                short_entry.update({
                    "type": "short",
                    "src_text": item.get("src_short"),
                    "tgt_text": tgt_short,
                    "model": trans_content.get("short_model"),
                    "scores": trans_content.get("short_scores")
                })
                lang_buffers[lang]['short'].append(short_entry)
                
                # 📊 统计长度 (按空格分割计算词数，中文等无空格语言可能需要特殊处理，此处为通用近似)
                len_stats[lang]['short'] += len(tgt_short.split())

            # --- 处理 Long Caption ---
            tgt_long = trans_content.get("long_translation")
            if tgt_long:
                long_entry = base_meta.copy()
                long_entry.update({
                    "type": "long",
                    "src_text": item.get("src_long"),
                    "tgt_text": tgt_long,
                    "model": trans_content.get("long_model"),
                    "scores": trans_content.get("long_scores")
                })
                lang_buffers[lang]['long'].append(long_entry)
                
                # 📊 统计长度
                len_stats[lang]['long'] += len(tgt_long.split())

    # 5. 保存文件并打印统计表
    print("\n💾 保存结果统计:")
    print("=" * 85)
    # 调整表头，增加 Avg Len 列
    print(f"{'Language':<12} | {'Type':<6} | {'Count':<8} | {'Avg Len':<8} | {'Output Filename'}")
    print("-" * 85)

    if not lang_buffers:
        print("⚠️ 警告: 没有提取到任何有效数据！")
        return

    # 按语言字母顺序排序输出
    for lang in sorted(lang_buffers.keys()):
        types = lang_buffers[lang]
        
        # --- 保存 Short 文件 ---
        if types['short']:
            count = len(types['short'])
            # 计算平均长度
            avg_len = len_stats[lang]['short'] / count if count > 0 else 0
            
            filename_s = f"{lang}_short.json"
            path_s = os.path.join(output_dir, filename_s)
            
            with open(path_s, 'w', encoding='utf-8') as f:
                json.dump(types['short'], f, ensure_ascii=False, indent=2)
            
            print(f"{lang:<12} | {'Short':<6} | {count:<8} | {avg_len:<8.1f} | {filename_s}")

        # --- 保存 Long 文件 ---
        if types['long']:
            count = len(types['long'])
            # 计算平均长度
            avg_len = len_stats[lang]['long'] / count if count > 0 else 0
            
            filename_l = f"{lang}_long.json"
            path_l = os.path.join(output_dir, filename_l)
            
            with open(path_l, 'w', encoding='utf-8') as f:
                json.dump(types['long'], f, ensure_ascii=False, indent=2)
            
            print(f"{lang:<12} | {'Long':<6} | {count:<8} | {avg_len:<8.1f} | {filename_l}")

    print("=" * 85)
    print(f"🎉 拆分完成！文件位于 '{output_dir}/' 目录下。")
    print(f"💡 注: 'Avg Len' 是基于空格分割的近似词数统计。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset and calc stats.")
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dataset = load_data(args.input_file)
    split_by_language_and_length(dataset, args.output_dir)
