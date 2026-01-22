import json
import os
import argparse
from collections import defaultdict
from tqdm import tqdm

# ==========================================
# 🛠️ 默认配置
# ==========================================
DEFAULT_INPUT_FILE = "dataset_optimal_filtered.json"
DEFAULT_OUTPUT_DIR = "final_datasets_split"

def load_data(file_path):
    print(f"📖 正在读取数据集: {file_path} ...")
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        exit(1)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        exit(1)

def split_by_language_and_length(data, output_dir):
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 已创建输出目录: {output_dir}")

    # 2. 初始化缓存容器
    # 结构: buffers[lang]['short'] = [], buffers[lang]['long'] = []
    lang_buffers = defaultdict(lambda: {'short': [], 'long': []})
    
    print("🚀 正在执行 [语言 + 长短句] 二级拆分...")
    
    # 3. 遍历数据
    stats = defaultdict(int)

    for item in tqdm(data):
        if 'translations' not in item: continue
        
        # 基础元数据 (图片路径等)
        base_meta = {
            "image_id": item.get("image_id"),
            "path": item.get("path"),
            # 原有的 wnid, width 等也可以带上，按需
        }

        # 遍历该条目下的所有语言
        for lang, trans_content in item['translations'].items():
            if not trans_content: continue

            # --- 处理 Short Caption ---
            # 只有当 short_translation 存在且不为空时才保存
            if trans_content.get("short_translation"):
                short_entry = base_meta.copy()
                short_entry.update({
                    "type": "short",
                    "src_text": item.get("src_short"),       # 统一字段名：源文本
                    "tgt_text": trans_content["short_translation"], # 统一字段名：目标文本
                    "model": trans_content.get("short_model"),
                    "scores": trans_content.get("short_scores")
                })
                lang_buffers[lang]['short'].append(short_entry)
                stats[f"{lang}_short"] += 1

            # --- 处理 Long Caption ---
            # 只有当 long_translation 存在且不为空时才保存
            if trans_content.get("long_translation"):
                long_entry = base_meta.copy()
                long_entry.update({
                    "type": "long",
                    "src_text": item.get("src_long"),        # 统一字段名：源文本
                    "tgt_text": trans_content["long_translation"],  # 统一字段名：目标文本
                    "model": trans_content.get("long_model"),
                    "scores": trans_content.get("long_scores")
                })
                lang_buffers[lang]['long'].append(long_entry)
                stats[f"{lang}_long"] += 1

    # 4. 保存文件
    print("\n💾 保存结果统计:")
    print("=" * 65)
    print(f"{'Language':<10} | {'Type':<6} | {'Count':<8} | {'Output Filename'}")
    print("-" * 65)

    if not lang_buffers:
        print("⚠️ 警告: 没有提取到任何有效数据！")
        return

    for lang, types in lang_buffers.items():
        # 保存 Short 文件
        if types['short']:
            filename_s = f"{lang}_short.json"
            path_s = os.path.join(output_dir, filename_s)
            with open(path_s, 'w', encoding='utf-8') as f:
                json.dump(types['short'], f, ensure_ascii=False, indent=2)
            print(f"{lang:<10} | {'Short':<6} | {len(types['short']):<8} | {filename_s}")

        # 保存 Long 文件
        if types['long']:
            filename_l = f"{lang}_long.json"
            path_l = os.path.join(output_dir, filename_l)
            with open(path_l, 'w', encoding='utf-8') as f:
                json.dump(types['long'], f, ensure_ascii=False, indent=2)
            print(f"{lang:<10} | {'Long':<6} | {len(types['long']):<8} | {filename_l}")

    print("=" * 65)
    print(f"🎉 拆分完成！文件位于 '{output_dir}/' 目录下。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into Language + Short/Long pairs.")
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dataset = load_data(args.input_file)
    split_by_language_and_length(dataset, args.output_dir)