import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建球的参数
theta = np.linspace(0, np.pi, 50)  # 纵向角度（0 到 π）
phi = np.linspace(0, 2 * np.pi, 50)  # 横向角度（0 到 2π）
theta, phi = np.meshgrid(theta, phi)

# 球体的坐标
r = 1
X = r * np.sin(theta) * np.cos(phi)
Y = r * np.sin(theta) * np.sin(phi)
Z = r * np.cos(theta)

# 创建 3D 图形
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# 画球体表面
ax.plot_surface(X, Y, Z, color='#FF8000', edgecolor='none', alpha=1.0)

# 设定视角
ax.view_init(elev=20, azim=30)

# **去掉坐标轴**
ax.set_xticks([])  # 去掉 x 轴刻度
ax.set_yticks([])  # 去掉 y 轴刻度
ax.set_zticks([])  # 去掉 z 轴刻度
ax.axis('off')  # 彻底去掉所有坐标轴

# 显示图像
plt.show()
