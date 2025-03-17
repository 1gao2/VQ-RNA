import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

# 数据准备
datasets = ['Am', 'Cm', 'Gm', 'Um', 'm1A', 'm5C', 'm5U', 'm6A', 'm6Am', 'Ψ']
methods = {
    'VQ-MethyINet':   [0.8924, 0.8702, 0.9259, 0.8120, 0.8752, 0.8492, 0.9283, 0.8454, 0.8593, 0.7469],
    'Moss-m7g':       [0.8667, 0.8463, 0.8864, 0.7955, 0.8757, 0.8337, 0.8966, 0.8265, 0.8415, 0.7252],
    'CNN-Predictor':  [0.8497, 0.8282, 0.8517, 0.7951, 0.8416, 0.8069, 0.8687, 0.8281, 0.8237, 0.6692],
    'GRU-Predictor':  [0.8396, 0.7995, 0.5932, 0.7756, 0.8750, 0.7710, 0.8766, 0.8725, 0.8008, 0.6880],
    'LSTM-Predictor': [0.6679, 0.6835, 0.5265, 0.6898, 0.8780, 0.7554, 0.8444, 0.8710, 0.8017, 0.6583],
    'TF-Predictor':   [0.8623, 0.8059, 0.8592, 0.7307, 0.8484, 0.8374, 0.8714, 0.7867, 0.8203, 0.7061]
}
# 颜色列表
colors = ['#92A5D1', '#C5DFF4', '#AEB2D1', '#D9B9D4', '#7C9895', '#C9DCC4']
# 创建图表
plt.figure(figsize=(10, 6))

# 遍历方法，绘制折线
for idx, (method, performance) in enumerate(methods.items()):
    plt.plot(datasets, performance, marker='o', label=method, color=colors[idx])

# 图表细节
plt.ylabel('Accuracy(ACC)', fontsize=12)
plt.ylim(0.65, 0.95)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Methods", fontsize=10, loc='lower left')
plt.tight_layout()
plt.savefig('ACC_Brokenline_comparison.svg', format='svg')
# 显示图表
plt.show()