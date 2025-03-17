import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

# 数据准备
datasets = ['32', '128', '512', '1024', '2048']
methods = {
    'ACC':   [0.8417, 0.8425, 0.8605, 0.8437, 0.8446],
    'SE':    [0.8192, 0.8169, 0.8420, 0.8206, 0.8209],
    'SP':    [0.8638, 0.8678, 0.8786, 0.8663, 0.8678],
    'AUC':   [0.9051, 0.9051, 0.9172, 0.9064, 0.9097],
    'MCC':   [0.6854, 0.6875, 0.7227, 0.6895, 0.6918]
}

# 颜色列表
colors = ['#92A5D1', '#C5DFF4', '#AEB2D1', '#D9B9D4', '#7C9895']
#colors = ['#92A5D1', '#C5DFF4', '#AEB2D1', '#D9B9D4', '#7C9895', '#C9DCC4']
# 创建图表
plt.figure(figsize=(10, 6))

# 遍历方法，绘制折线
for idx, (method, performance) in enumerate(methods.items()):
    plt.plot(datasets, performance, marker='o', label=method, color=colors[idx])

# 图表细节
plt.xlabel('Size of codebook(K)', fontsize=12)
plt.ylabel('Performances', fontsize=12)
plt.ylim(0.6, 0.95 )
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Metrics", fontsize=10, loc='lower left')
plt.tight_layout()
plt.savefig('vqsize_Brokenline_comparison.svg', format='svg')
# 显示图表
plt.show()