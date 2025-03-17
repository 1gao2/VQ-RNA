import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

# 数据准备
datasets = ['Am', 'Cm', 'Gm', 'Um', 'm1A', 'm5C', 'm5U', 'm6A', 'm6Am', 'Ψ']
methods = {
    'VQ-MethyINet':   [0.9409, 0.9201, 0.9569, 0.8717, 0.9374, 0.9153, 0.9674, 0.9217, 0.9236, 0.8168],
    'Moss-m7g':       [0.9302, 0.9155, 0.9375, 0.8504, 0.9414, 0.8822, 0.9484, 0.9049, 0.9147, 0.8163],
    'CNN-Predictor':  [0.9104, 0.8945, 0.9171, 0.8635, 0.9146, 0.8749, 0.9361, 0.9035, 0.8810, 0.7349],
    'GRU-Predictor':  [0.9047, 0.8802, 0.6310, 0.8450, 0.9422, 0.8438, 0.9333, 0.9411, 0.8567, 0.7677],
    'LSTM-Predictor': [0.7324, 0.7267, 0.5384, 0.7622, 0.9399, 0.8229, 0.9094, 0.9413, 0.8584, 0.6951],
    'TF-Predictor':   [0.9197, 0.8857, 0.9170, 0.8179, 0.9219, 0.9074, 0.9434, 0.8690, 0.8978, 0.7721]
}

# 颜色列表
colors = ['#92A5D1', '#C5DFF4', '#AEB2D1', '#D9B9D4', '#7C9895', '#C9DCC4']
# 创建图表
plt.figure(figsize=(10, 6))

# 遍历方法，绘制折线
for idx, (method, performance) in enumerate(methods.items()):
    plt.plot(datasets, performance, marker='o', label=method, color=colors[idx])

# 图表细节
plt.ylabel('AUC', fontsize=12)
plt.ylim(0.7, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Methods", fontsize=10, loc='lower left')
plt.tight_layout()
plt.savefig('AUC_Brokenline_comparison.svg', format='svg')
# 显示图表
plt.show()