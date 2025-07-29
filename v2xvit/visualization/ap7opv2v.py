import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 原始数据
data_points = {
    "Ours": np.array([
        [0, 0.7341], [0.1, 0.7193], [0.2, 0.6948], [0.3, 0.6789],
        [0.4, 0.6512], [0.5, 0.6340], [0.6, 0.6102], [0.7, 0.5911],
        [0.8, 0.5801], [0.9, 0.5892], [1, 0.5783]
    ]),
    "How2comm": np.array([
        [0, 0.7110], [0.1, 0.7063], [0.2, 0.6988], [0.3, 0.6303],
        [0.4, 0.6370], [0.5, 0.6240], [0.6, 0.5700], [0.7, 0.5686],
        [0.8, 0.5717], [0.9, 0.5733], [1, 0.5433]
    ]),
    "Where2comm": np.array([
        [0, 0.6629], [0.1, 0.6462], [0.2, 0.6254], [0.3, 0.6019],
        [0.4, 0.5895], [0.5, 0.5624], [0.6, 0.5501], [0.7, 0.5298],
        [0.8, 0.5209], [0.9, 0.5188], [1, 0.5206]
    ]),
    "No Fusion": np.array([
        [0, 0.4866], [0.1, 0.4866], [0.2, 0.4866], [0.3, 0.4866],
        [0.4, 0.4866], [0.5, 0.4866], [0.6, 0.4866], [0.7, 0.4866],
        [0.8, 0.4866], [0.9, 0.4866], [1, 0.4866]
    ]),
    "V2VNet": np.array([
        [0, 0.6329], [0.1, 0.6062], [0.2, 0.5754], [0.3, 0.5486],
        [0.4, 0.5135], [0.5, 0.4964], [0.6, 0.4701], [0.7, 0.4428],
        [0.8, 0.4159], [0.9, 0.4108], [1, 0.4006]
    ])
}

# 2. 设置美学风格
sns.set_style("whitegrid")
# 使用更专业的调色板，例如 'colorblind' 或 'deep'
# 'colorblind' 对色弱友好，'deep' 颜色饱和度适中
palette = sns.color_palette("deep", n_colors=5)
markers = ['o', 's', 'D', '^', 'p'] # 圆形, 方形, 菱形, 三角形
linestyles = ['-', '--', '-.', ':', '--']

# 3. 创建图形 (关键：调整figsize来拉伸横轴)
# 将宽度设置得比高度大得多，例如 14x7
plt.figure(figsize=(14, 7))

# 4. 循环绘图，使代码更简洁
for i, (label, data) in enumerate(data_points.items()):
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y,
             marker=markers[i],         # 使用不同的标记
             markersize=8,
             linestyle=linestyles[i],   # 使用不同的线型
             color=palette[i],          # 使用调色板中的颜色
             lw=2.5,                    # 稍微加粗线条
             label=label)

# 5. 美化图表
# 使用更清晰的字体大小和粗细
plt.xlabel('Transmission Delay (s)', fontsize=16, fontweight='bold', labelpad=15)
plt.ylabel('AP@0.7 on OPV2V', fontsize=16, fontweight='bold', labelpad=15)

# 调整坐标轴刻度字体
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=14)
plt.yticks(fontsize=14)

# 设置Y轴范围，留出上下空间
y_min = min(arr[:, 1].min() for arr in data_points.values())
y_max = max(arr[:, 1].max() for arr in data_points.values())
plt.ylim(y_min - 0.05, y_max + 0.05)
plt.xlim(-0.05, 1.05)

# 添加更专业的图例
# 'loc'='best' 会自动寻找最佳位置，也可以手动指定如 'upper right'
# 'ncol' 可以将图例分为多列，如果标签很长的话
legend = plt.legend(fontsize=14, frameon=True, shadow=True, title='Models', title_fontsize='16')
legend.get_title().set_fontweight('bold')


# 添加网格线，并设置样式
plt.grid(True, which='both', linestyle='--', linewidth=0.7)

# 移除顶部和右侧的边框
sns.despine()

# 自动调整布局
plt.tight_layout()

# 显示图表
plt.show()

