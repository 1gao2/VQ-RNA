import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['svg.fonttype'] = 'none'
# 设置全局字体和字号
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20

# 数据
methods = ['VQ-MethylNet', 'w/o recon_loss', 'w/o VQ_loss', 'w/o VQ_loss and recon_loss']
attributes = ['ACC', 'AUC', 'MCC']
values = [
    [0.8605, 0.9172, 0.7227],  # VQ-MethylNet
    [0.8431, 0.9048, 0.6882],  # w/o recon_loss   loss1+loss2
    [0.5760, 0.6268, 0.1541],  # w/o VQ_loss   loss1+loss3
    [0.5756, 0.6563, 0.1541]   # w/o VQ_loss and recon_loss  loss1
]

colors = ['#92A5D1', '#C5DFF4', '#AEB2D1', '#D9B9D4']

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
ax.set_title('Loss Ablation Analysis')
ax.set_xticks(x + width * (len(methods) - 1) / 2)  # 调整x轴刻度位置
ax.set_xticklabels(attributes)
ax.legend(fontsize=16, loc='upper left')

# 保存图表
plt.savefig('vq_loss_ablation.svg', format='svg')

# 显示图表
plt.show()
