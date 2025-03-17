import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体和字号
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20

# 数据
methods = ['VQ-MethylNet', 'w/o VQ', 'w/o Transformer', 'Encoder with one_CNNLayer', 'Encoder with two_CNNLayers']
attributes = ['ACC', 'SE', 'SP', 'AUC', 'MCC']
values = [
    [0.8605, 0.8420, 0.8786, 0.9172, 0.7227],  # VQ-MethylNet
    [0.8433, 0.8164, 0.8698, 0.9073, 0.6888],  # w/o VQ
    [0.8106, 0.7896, 0.8314, 0.8684, 0.6224],  # w/o Transformer
    [0.7805, 0.7418, 0.8185, 0.8491, 0.5642],  # CNN_oneLayer
    [0.8167, 0.7865, 0.8462, 0.8884, 0.6362]   # CNN_twoLayer
]

colors = ['#92A5D1', '#C5DFF4', '#C9DCC4', '#AEB2D1', '#D9B9D4']
# 转置数据以按属性分组
values = np.array(values).T  # 转置为 (5, 6)

# 设置柱状图参数
x = np.arange(len(attributes))  # 属性的x轴位置
width = 0.15  # 柱的宽度

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制分组柱状图
for i, method in enumerate(methods):
    ax.bar(x + i * width, values[:, i], width, label=method, color=colors[i])

# 设置y轴范围
ax.set_ylim(0, 1)

# 添加标签和标题
ax.set_xlabel('Attributes')
ax.set_ylabel('Performances (%)')
ax.set_title('Component Ablation Analysis')
ax.set_xticks(x + width * (len(methods) - 1) / 2)  # 调整x轴刻度位置
ax.set_xticklabels(attributes)
ax.legend(fontsize=16, loc='upper left')

# 保存图表
plt.savefig('Component_Ablation.svg', format='svg')

# 显示图表
plt.show()
