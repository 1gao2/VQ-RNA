import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

# 数据准备
datasets = ['Am', 'Cm', 'Gm', 'Um', 'm1A', 'm5C', 'm5U', 'm6A', 'm6Am', 'Ψ']
methods = {
    'VQ-MethyINet':   [0.7875, 0.7408, 0.8526, 0.6255, 0.7506, 0.6987, 0.8572, 0.6912, 0.7281, 0.4945],
    'Moss-m7g':       [0.7353, 0.6946, 0.7738, 0.5966, 0.7516, 0.6680, 0.7948, 0.6534, 0.6904, 0.4629],
    'CNN-Predictor':  [0.7002, 0.6574, 0.7042, 0.5913, 0.6835, 0.6138, 0.7377, 0.6563, 0.6570, 0.3400],
    'GRU-Predictor':  [0.6796, 0.5997, 0.1868, 0.5519, 0.7502, 0.5433, 0.7537, 0.7451, 0.6248, 0.3845],
    'LSTM-Predictor': [0.3411, 0.3686, 0.0532, 0.3858, 0.7563, 0.5133, 0.6913, 0.7421, 0.6253, 0.3326],
    'TF-Predictor':   [0.7276, 0.6133, 0.7201, 0.4631, 0.6972, 0.6749, 0.7453, 0.5739, 0.6466, 0.4150]
}

# 颜色列表
colors = ['#92A5D1', '#C5DFF4', '#AEB2D1', '#D9B9D4', '#7C9895', '#C9DCC4']
# 创建图表
plt.figure(figsize=(10, 6))

# 遍历方法，绘制折线
for idx, (method, performance) in enumerate(methods.items()):
    plt.plot(datasets, performance, marker='o', label=method, color=colors[idx])

# 图表细节
plt.ylabel('MCC', fontsize=12)
plt.ylim(0.3, 0.9 )
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Methods", fontsize=10, loc='lower left')
plt.tight_layout()
plt.savefig('MCC_Brokenline_comparison.svg', format='svg')
# 显示图表
plt.show()