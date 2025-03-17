import numpy as np
import matplotlib.pyplot as plt

# 半径
r = 0.5

# 定义角度范围
# 为了保证 x, y, z 都为正，phi 和 theta 均取 [0, π/2]
phi = np.linspace(0, np.pi/2, 50)     # 极角
theta = np.linspace(0, np.pi/2, 50)     # 方位角

phi, theta = np.meshgrid(phi, theta)

# 球面参数方程
x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)

# 绘制
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='#FF8000', alpha=0.7, shade=False)

# 调整视角
ax.view_init(elev=20, azim=300)
# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 保存为 SVG
plt.savefig("3d_chart.svg", format="svg")

# 显示图像
plt.show()
