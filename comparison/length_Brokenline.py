import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

# 数据准备
datasets = ['51', '201', '301', '401', '501', '701', '901', '10001']
methods = {
    'ACC':   [0.8231, 0.8361, 0.8404, 0.8448, 0.8605, 0.8259, 0.8416, 0.8390],
    'SE':    [0.7978, 0.8070, 0.8162, 0.8196, 0.8420, 0.7962, 0.8180, 0.8172],
    'SP':    [0.8481, 0.8647, 0.8643, 0.8696, 0.8786, 0.8553, 0.8647, 0.8604],
    'AUC':   [0.8882, 0.8992, 0.9042, 0.9090, 0.9172, 0.8885, 0.9070, 0.9039],
    'MCC':   [0.6489, 0.6750, 0.6832, 0.6920, 0.7227, 0.6547, 0.6852, 0.6801]

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
plt.xlabel('Length(bp)', fontsize=12)
plt.ylabel('Performances', fontsize=12)
plt.ylim(0.6, 0.95 )
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Metrics", fontsize=10, loc='lower left')
plt.tight_layout()
plt.savefig('length_Brokenline_comparison.svg', format='svg')
# 显示图表
plt.show()