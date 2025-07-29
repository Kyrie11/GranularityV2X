import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 原始数据
data_points = {
    "Ours": np.array([
        [0, 0.5452], [0.1, 0.5404], [0.2, 0.5361], [0.3, 0.5288],
        [0.4, 0.5242], [0.5, 0.5201], [0.6, 0.5158], [0.7, 0.5105],
        [0.8, 0.5068], [0.9, 0.5018], [1, 0.4951]
    ]),
    "How2comm": np.array([
        [0, 0.5218], [0.1, 0.5103], [0.2, 0.4962], [0.3, 0.4893],
        [0.4, 0.4732], [0.5, 0.4341], [0.6, 0.4398], [0.7, 0.4325],
        [0.8, 0.4258], [0.9, 0.4208], [1, 0.4161]
    ]),
    "Where2comm": np.array([
        [0, 0.5229], [0.1, 0.5132], [0.2, 0.5114], [0.3, 0.4989],
        [0.4, 0.4765], [0.5, 0.4704], [0.6, 0.4671], [0.7, 0.4628],
        [0.8, 0.4709], [0.9, 0.4688], [1, 0.4606]
    ]),
    "No Fusion": np.array([
        [0, 0.4357], [0.1, 0.4357], [0.2, 0.4357], [0.3, 0.4357],
        [0.4, 0.4357], [0.5, 0.4357], [0.6, 0.4357], [0.7, 0.4357],
        [0.8, 0.4357], [0.9, 0.4357], [1, 0.4357]
    ]),
    "V2VNet": np.array([
        [0, 0.4571], [0.1, 0.4536], [0.2, 0.4482], [0.3, 0.4463],
        [0.4, 0.4425], [0.5, 0.4383], [0.6, 0.4351], [0.7, 0.4307],
        [0.8, 0.4279], [0.9, 0.4358], [1, 0.4356]
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

