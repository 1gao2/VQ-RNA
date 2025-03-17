import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体和字号
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20

# 数据
methods = ['VQ-MethylNet', 'Moss-m7g', 'CNN', 'GRU', 'LSTM', 'Transformer']
attributes = ['ACC', 'SE', 'SP', 'MCC', 'AUC']
values = [
    [0.8605, 0.8420, 0.8786, 0.7227, 0.9172],  # VQ-MethylNet
    [0.8394, 0.8122, 0.8662, 0.6821, 0.9041],  # Moss-m7g
    [0.8163, 0.8000, 0.8321, 0.6341, 0.8830],  # CNN
    [0.7892, 0.7685, 0.8092, 0.5820, 0.8546],  # GRU
    [0.7376, 0.6964, 0.7781, 0.4810, 0.7927],  # LSTM
    [0.8128, 0.7994, 0.8259, 0.6277, 0.8852]   # Transformer
]

colors = ['#82A7D1', '#B6D7E9', '#DBEAF3', '#FEE8DD', '#F5B99E', '#E2745E']

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
ax.set_title('VQ-DNA Comparison Analysis')
ax.set_xticks(x + width * (len(methods) - 1) / 2)  # 调整x轴刻度位置
ax.set_xticklabels(attributes)
ax.legend(fontsize=16, loc='upper left')

# 保存图表
plt.savefig('vq_comparison.svg', format='svg')

# 显示图表
plt.show()
