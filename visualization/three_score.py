import matplotlib.pyplot as plt
import numpy as np

# === 1. 设置学术论文风格 (Times New Roman) ===
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
# [修改点] 基础字号从 12 提升到 15
plt.rcParams['font.size'] = 15

# === 2. 准备数据 ===
languages = ['Uyghur', 'Kazakh', 'Kyrgyz', 'Tajik', 'Uzbek', 'Urdu', 'Avg']
qe_scores = [82.1, 83.3, 82.8, 80.5, 81.8, 80.3, 82.4]
bs_scores = [93.87, 93.85, 93.58, 95.49, 94.71, 95.84, 94.2]
clip_scores = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

x = np.arange(len(languages))
width = 0.25

# === 3. 创建画布 ===
# 保持尺寸不变，增大字体占比
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# === 4. 绘制柱子 (保持颜色和样式) ===
rects1 = ax1.bar(x - width, qe_scores, width, label='QE (Text)', color='#4ad2ff', edgecolor='white', zorder=3)
rects2 = ax1.bar(x, bs_scores, width, label='BS (Semantic)', color='#ffab40', edgecolor='white', zorder=3)
rects3 = ax2.bar(x + width, clip_scores, width, label='CLIP (Visual)', color='#2e7d32', edgecolor='white', zorder=3)

# === 5. 设置坐标轴和标签 (大幅增大字号) ===
# [修改点] 轴标题字号 14 -> 18 (加粗)
ax1.set_ylabel('Text Metrics (Score)', fontsize=18, labelpad=10)
ax1.set_ylim(0, 125) # 稍微增加顶部空间以容纳更大的图例
# [修改点] X轴标签字号 12 -> 16
ax1.set_xticks(x)
ax1.set_xticklabels(languages, fontsize=16)
# [修改点] Y轴刻度字号 11 -> 15
ax1.tick_params(axis='y', labelsize=15)

# 右轴设置
# [修改点] 右轴标题字号 14 -> 18 (加粗)
ax2.set_ylabel('Visual Metric (Score)', fontsize=18, labelpad=10)
ax2.set_ylim(0, 0.48)
ax2.tick_params(axis='y', labelsize=15)

# === 6. 添加数值标签 (增大字号并加粗) ===
def autolabel(rects, ax, is_float=False):
    for rect in rects:
        height = rect.get_height()
        label = f'{height:.2f}' if is_float else f'{height:.1f}'
        # [修改点] 数值标签字号 9 -> 11，且加粗 (fontweight='bold')
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=16, rotation=90)

autolabel(rects1, ax1)
autolabel(rects2, ax1)
autolabel(rects3, ax2, is_float=True)

# === 7. 合并图例 (增大字号并调整位置) ===
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# [修改点] 图例字号 12 -> 15
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
           bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=15)

# === 8. 调整布局 ===
plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
# [修改点] 标题字号 14 -> 20
plt.title('Multidimensional Quality Analysis', y=1.15, fontsize=20)
plt.tight_layout()

# 保存
plt.savefig('quality_analysis_chart.pdf', dpi=300, bbox_inches='tight')
plt.show()