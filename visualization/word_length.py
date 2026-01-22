# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from scipy.stats import gaussian_kde
#
# # ==========================================
# # 1. Load Data (读取真实 JSON 文件)
# # ==========================================
# INPUT_FILE = "dataset_optimal_filtered.json"
#
# print(f"📖 正在读取文件: {INPUT_FILE} ...")
#
# if not os.path.exists(INPUT_FILE):
#     print(f"❌ 错误: 找不到文件 {INPUT_FILE}")
#     print("请确保 json 文件与脚本在同一目录下。")
#     exit(1)
#
# try:
#     with open(INPUT_FILE, 'r', encoding='utf-8') as f:
#         # 尝试读取，兼容 JSON Array 和 JSONL
#         first_char = f.read(1)
#         f.seek(0)
#         if first_char == '[':
#             data = json.load(f)
#         else:
#             data = [json.loads(line) for line in f if line.strip()]
#     print(f"✅ 成功加载 {len(data)} 条数据。")
# except Exception as e:
#     print(f"❌ 读取 JSON 失败: {e}")
#     exit(1)
#
# # ==========================================
# # 2. Extract Lengths (提取长度)
# # ==========================================
# short_lengths = []
# long_lengths = []
#
# print("⚙️ 正在统计句子长度...")
# for item in data:
#     if 'translations' not in item:
#         continue
#
#     for lang, content in item['translations'].items():
#         # Process Short Captions
#         if 'short_translation' in content and content['short_translation']:
#             text = content['short_translation']
#             # Calculate length by splitting on whitespace (approximate word count)
#             # 针对中文等无空格语言，len(split)可能为1，这是预期行为（统计为1句）
#             # 如果需要更精细，可以根据语言判断
#             length = len(text.split())
#             short_lengths.append(length)
#
#         # Process Long Captions
#         if 'long_translation' in content and content['long_translation']:
#             text = content['long_translation']
#             length = len(text.split())
#             long_lengths.append(length)
#
# print(f"📊 统计完成: Short样本数={len(short_lengths)}, Long样本数={len(long_lengths)}")
#
# # 转为 Numpy 数组，确保绘图安全
# short_data = np.array(short_lengths)
# long_data = np.array(long_lengths)
#
# # ==========================================
# # 3. Plot Histogram (纯 Matplotlib 安全版)
# # ==========================================
# print("🎨 正在绘图...")
#
# # 设置画板
# fig, ax = plt.subplots(figsize=(8, 5))
#
# # ----------------------------------------
# # 绘制 Short Caption (蓝色)
# # ----------------------------------------
# if len(short_data) > 0:
#     # 1. 绘制直方图 (Histogram)
#     ax.hist(short_data, bins=15, density=True, alpha=0.5,
#             color='#3498db', label='Short Caption', edgecolor='white')
#
#     # 2. 绘制 KDE 曲线 (手动计算，避开 Seaborn 错误)
#     try:
#         density_short = gaussian_kde(short_data)
#         # 生成 X 轴坐标点
#         xs_short = np.linspace(0, max(short_data) * 1.2, 200)
#         # 绘制曲线
#         ax.plot(xs_short, density_short(xs_short), color='#3498db', linewidth=2)
#     except Exception as e:
#         print(f"⚠️ Short KDE 绘制失败 (数据可能太少): {e}")
#
# # ----------------------------------------
# # 绘制 Long Caption (橙色)
# # ----------------------------------------
# if len(long_data) > 0:
#     # 1. 绘制直方图
#     ax.hist(long_data, bins=25, density=True, alpha=0.5,
#             color='#e67e22', label='Long Caption', edgecolor='white')
#
#     # 2. 绘制 KDE 曲线
#     try:
#         density_long = gaussian_kde(long_data)
#         xs_long = np.linspace(0, max(long_data) * 1.2, 200)
#         ax.plot(xs_long, density_long(xs_long), color='#e67e22', linewidth=2)
#     except Exception as e:
#         print(f"⚠️ Long KDE 绘制失败: {e}")
#
# # ----------------------------------------
# # 美化图表 (ACL 风格) - 字体加大版
# # ----------------------------------------
# # Increased font sizes for better visibility in papers
# ax.set_xlabel("Sentence Length (Number of Tokens)", fontsize=16)   # fontweight='bold'
# ax.set_ylabel("Density", fontsize=16)              # fontweight='bold'
# ax.set_xlim(0, 80)
# ax.set_title("Sentence Length Distribution: Dual-Granularity", fontsize=18, pad=15)
# ax.legend(fontsize=14, loc='upper right')
#
# # Increase tick label size
# ax.tick_params(axis='both', which='major', labelsize=14)
#
# ax.grid(axis='y', linestyle='--', alpha=0.5)
#
# # 调整布局并保存
# plt.tight_layout()
#
# # 保存
# try:
#     plt.savefig("fig_len_dist.pdf", dpi=300)
#     # plt.savefig("fig_len_dist.png", dpi=300)
#     print("✅ 图片已生成: fig_len_dist.png / pdf")
# except Exception as e:
#     print(f"❌ 保存图片失败: {e}")
#
# # 显示图片 (如果在服务器无图形界面运行，这行可能会报错，可注释掉)
# try:
#     plt.show()
# except:
#     pass

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import defaultdict
from scipy.stats import gaussian_kde

# ==========================================
# 1. Load Data (读取真实 JSON 文件)
# ==========================================
INPUT_FILE = "dataset_optimal_filtered.json"

print(f"📖 正在读取文件: {INPUT_FILE} ...")

if not os.path.exists(INPUT_FILE):
    print(f"❌ 错误: 找不到文件 {INPUT_FILE}")
    print("请确保 json 文件与脚本在同一目录下。")
    exit(1)

try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    print(f"✅ 成功加载 {len(data)} 条数据。")
except Exception as e:
    print(f"❌ 读取 JSON 失败: {e}")
    exit(1)

# ==========================================
# 2. Extract Lengths & Statistics (提取长度并统计)
# ==========================================
# 全局列表 (用于画图)
global_short_lengths = []
global_long_lengths = []

# 分语言统计字典: stats[lang]['short'] = [len1, len2...]
lang_stats = defaultdict(lambda: {'short': [], 'long': []})

print("⚙️ 正在统计句子长度...")
for item in data:
    if 'translations' not in item:
        continue

    for lang, content in item['translations'].items():
        # 统一语言键名 (小写)
        lang_key = lang.lower()

        # Process Short Captions
        if 'short_translation' in content and content['short_translation']:
            text = content['short_translation']
            length = len(text.strip().split())  # 按空格分词统计长度

            lang_stats[lang_key]['short'].append(length)
            global_short_lengths.append(length)

        # Process Long Captions
        if 'long_translation' in content and content['long_translation']:
            text = content['long_translation']
            length = len(text.strip().split())

            lang_stats[lang_key]['long'].append(length)
            global_long_lengths.append(length)

print(f"📊 统计完成: Short总样本数={len(global_short_lengths)}, Long总样本数={len(global_long_lengths)}")

# ==========================================
# 3. Output Table (输出表格)
# ==========================================
print("\n" + "=" * 60)
print("📊 平均长度统计 (Average Sentence Length)")
print("=" * 60)

table_rows = []
custom_order = ["uyghur", "kazakh", "kyrgyz", "tajik", "uzbek", "urdu"]
sorted_langs = sorted(lang_stats.keys())  # 或者使用 custom_order 排序逻辑

# 1. 遍历各语言
for lang in sorted_langs:
    shorts = lang_stats[lang]['short']
    longs = lang_stats[lang]['long']

    avg_s = np.mean(shorts) if shorts else 0.0
    avg_l = np.mean(longs) if longs else 0.0

    table_rows.append({
        "Language": lang.capitalize(),
        "Avg Length (Short)": avg_s,
        "Avg Length (Long)": avg_l,
        "Count (Short)": len(shorts),
        "Count (Long)": len(longs)
    })

# 2. 添加全局平均行 (Average)
avg_row = {
    "Language": "AVERAGE",
    "Avg Length (Short)": np.mean(global_short_lengths) if global_short_lengths else 0.0,
    "Avg Length (Long)": np.mean(global_long_lengths) if global_long_lengths else 0.0,
    "Count (Short)": len(global_short_lengths),
    "Count (Long)": len(global_long_lengths)
}
table_rows.append(avg_row)

# 3. 生成 Pandas DataFrame 并打印
df = pd.DataFrame(table_rows)

# 设置显示格式：浮点数保留2位
pd.options.display.float_format = '{:.2f}'.format
# 设置列对齐和宽度
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(df.to_string(index=False))
print("=" * 60 + "\n")

# 可选：保存表格到 CSV
df.to_csv("length_statistics.csv", index=False)
print("✅ 表格数据已保存至 length_statistics.csv")

# ==========================================
# 4. Plot Histogram (绘图 - 保持不变)
# ==========================================
print("🎨 正在绘图...")

short_data = np.array(global_short_lengths)
long_data = np.array(global_long_lengths)

# 设置画板
fig, ax = plt.subplots(figsize=(8, 5))

# 绘制 Short Caption (蓝色)
if len(short_data) > 0:
    ax.hist(short_data, bins=15, density=True, alpha=0.5,
            color='#3498db', label='Short Caption', edgecolor='white')
    try:
        density_short = gaussian_kde(short_data)
        xs_short = np.linspace(0, max(short_data) * 1.2, 200)
        ax.plot(xs_short, density_short(xs_short), color='#3498db', linewidth=2)
    except Exception as e:
        print(f"⚠️ Short KDE 绘制失败: {e}")

# 绘制 Long Caption (橙色)
if len(long_data) > 0:
    ax.hist(long_data, bins=25, density=True, alpha=0.5,
            color='#e67e22', label='Long Caption', edgecolor='white')
    try:
        density_long = gaussian_kde(long_data)
        xs_long = np.linspace(0, max(long_data) * 1.2, 200)
        ax.plot(xs_long, density_long(xs_long), color='#e67e22', linewidth=2)
    except Exception as e:
        print(f"⚠️ Long KDE 绘制失败: {e}")

# 美化图表
ax.set_xlabel("Sentence Length (Number of Tokens)", fontsize=16)
ax.set_ylabel("Density", fontsize=16)
ax.set_xlim(0, 80)
ax.set_title("Sentence Length Distribution: Dual-Granularity", fontsize=18, pad=15)
ax.legend(fontsize=14, loc='upper right')
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

try:
    plt.savefig("fig_len_dist.pdf", dpi=300)
    print("✅ 图片已生成: fig_len_dist.pdf")
except Exception as e:
    print(f"❌ 保存图片失败: {e}")

# plt.show()