import torch
import matplotlib.pyplot as plt

# 假设 result 是一个 (1, 500) 的 tensor
result = torch.randn(1, 500)

# 将 result 转换为 numpy 数组
result_np = result.squeeze().numpy()

# 画出柱状图
plt.figure(figsize=(10, 6))
plt.bar(range(len(result_np)), result_np)  # 使用 bar() 绘制柱状图
plt.title('Bar Chart of Result')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()
