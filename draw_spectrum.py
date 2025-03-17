import pandas as pd
import matplotlib.pyplot as plt


tf_idf_matrix = pd.read_csv('tf_idf_matrix_sorted.csv')

# 创建一个新的图形窗口
plt.figure(figsize=(20, 60))

colors = ['#acafcd', '#90a1ca', '#c3dcf0', '#d6b7d1', '#C37BA4', '#c6d9c1', '#98df8a', '#8FD0D9', '#F6D997', '#ffb07d']

for i in range(10):
    plt.subplot(10, 1, i + 1)
    values = tf_idf_matrix.iloc[:, i]  # 取出第 i 个数据集的 TF-IDF 值
    plt.bar(range(len(values)), values, color=colors[i], label=f'Dataset_{i + 1}')

    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend().set_visible(False)
    plt.axis('off')

plt.tight_layout()
plt.savefig('Z-score.png')
plt.show()
plt.close()
