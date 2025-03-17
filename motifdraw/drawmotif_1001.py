
import pandas as pd
from tqdm import tqdm
import itertools
import numpy as np
import random
import copy
import time
import torch
tqdm.pandas(ascii=True)
import os
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from termcolor import colored
from motifmodel.motifmodel_1001 import Lucky
from tangermeme.plot import plot_logo
from tangermeme.ism import saturation_mutagenesis
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    'batch_size': 32,
    'data_index': 1,  # 示例文件索引
    'seq_len': 1001,    # 序列长度
    'seed': 2
}

def read_file(data_type, file_index):
    datas = pd.read_csv(f"../data/1-1.csv")
    seq =  list(datas['data'])
    label = list(datas['label'])

    seq = [s.replace(' ', '').replace('T', 'U') for s in seq]

    return seq, label

def encode_sequence_1mer(sequences, max_seq):
    k = 1
    overlap = False

    all_kmer = [''.join(p) for p in itertools.product(['A', 'C', 'G', 'U', '-'], repeat=k)]
    kmer_dict = {all_kmer[i]: i for i in range(len(all_kmer))}

    encoded_sequences = []
    if overlap:
        max_length = max_seq - k + 1

    else:
        max_length = max_seq // k

    for seq in sequences:
        encoded_seq = []
        start_site = len(seq) // 2 - max_length // 2
        for i in range(start_site, start_site + max_length, k):
            encoded_seq.append(kmer_dict[seq[i:i+k]])

        encoded_sequences.append(encoded_seq+[0]*(max_length-len(encoded_seq)))

    return np.array(encoded_sequences)

def load_model_and_data():
    """
    加载模型和验证数据。
    """
    # 加载数据
    test_x, test_y = read_file(data_type='test', file_index=params['data_index'])
    test_x, test_y = np.array(test_x), np.array(test_y)
    test_x = encode_sequence_1mer(test_x, max_seq=params['seq_len'])

    test_dataset = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, drop_last=True)

    # 加载模型
    model = Lucky().to(device)
    model.load_state_dict(torch.load(f"../save/length/length1001/data{params['data_index']}/seed{params['seed']}.pth")['state_dict'])

    return model, test_loader, test_dataset


class SliceWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SliceWrapper, self).__init__()
        self.model = model
        # self.target = target

    def forward(self, X):
        # print(self.model(X))
        # print(X.shape)
        out = self.model(X)
        out = out[0][:, 1].unsqueeze(1)
        # print(out.shape)
        result = torch.sigmoid(out)
        #print(result)
        return result


model, test_loader, test_dataset = load_model_and_data()
wrapper = SliceWrapper(model).cuda()

# 初始化累加的特征重要性矩阵
total_attr1 = None
total_attr2 = None
total_attr3 = None
total_attr4 = None
total_attr5 = None
total_attr6 = None
total_attr7 = None
total_attr8 = None
total_attr9 = None
total_attr10 = None
total_attr11 = None

# 遍历 test_dataset 的所有样本
for i in range(len(test_dataset)):
    # 获取第 i 个样本的特征
    x = test_dataset[i][0].unsqueeze(dim=0)  # 取第 i 个样本并扩展维度
    x = F.one_hot(x, num_classes=4).transpose(1, 2).float()  # 将序列转为 one-hot 编码

    # 计算特征重要性
    X_attr1 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=50 - 50, end=50 + 50, device='cpu')
    X_attr2 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=150 - 50, end=150 + 50, device='cpu')
    X_attr3 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=250 - 50, end=250 + 50, device='cpu')
    X_attr4 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=350 - 50, end=350 + 50, device='cpu')
    X_attr5 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=450 - 50, end=450 + 50, device='cpu')
    X_attr6 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=550 - 50, end=550 + 50, device='cpu')
    X_attr7 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=650 - 50, end=650 + 50, device='cpu')
    X_attr8 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=750 - 50, end=750 + 50, device='cpu')
    X_attr9 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=850 - 50, end=850 + 50, device='cpu')
    X_attr10 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=950 - 50, end=950 + 50, device='cpu')
    X_attr11 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=0, end=1000, device='cpu')

    # 累加特征重要性
    if total_attr1 is None:
        total_attr1 = X_attr1  # 初始化累加矩阵
    else:
        total_attr1 += X_attr1  # 累加特征重要性

    if total_attr2 is None:
        total_attr2 = X_attr2  # 初始化累加矩阵
    else:
        total_attr2 += X_attr2  # 累加特征重要性

    if total_attr3 is None:
        total_attr3 = X_attr3  # 初始化累加矩阵
    else:
        total_attr3 += X_attr3  # 累加特征重要性

    if total_attr4 is None:
        total_attr4 = X_attr4  # 初始化累加矩阵
    else:
        total_attr4 += X_attr4  # 累加特征重要性

    if total_attr5 is None:
        total_attr5 = X_attr5  # 初始化累加矩阵
    else:
        total_attr5 += X_attr5  # 累加特征重要性

    if total_attr6 is None:
        total_attr6 = X_attr6  # 初始化累加矩阵
    else:
        total_attr6 += X_attr6  # 累加特征重要性

    if total_attr7 is None:
        total_attr7 = X_attr7  # 初始化累加矩阵
    else:
        total_attr7 += X_attr7  # 累加特征重要性

    if total_attr8 is None:
        total_attr8 = X_attr8  # 初始化累加矩阵
    else:
        total_attr8 += X_attr8  # 累加特征重要性

    if total_attr9 is None:
        total_attr9 = X_attr9  # 初始化累加矩阵
    else:
        total_attr9 += X_attr9  # 累加特征重要性

    if total_attr10 is None:
        total_attr10 = X_attr10  # 初始化累加矩阵
    else:
        total_attr10 += X_attr10  # 累加特征重要性

    if total_attr11 is None:
        total_attr11 = X_attr11  # 初始化累加矩阵
    else:
        total_attr11 += X_attr11  # 累加特征重要性

# Feature importance score
avg_attr11 = abs(total_attr11) / len(test_dataset)
importance_score = torch.sum(avg_attr11, dim=1)
importance_score = importance_score.squeeze().numpy()

plt.figure(figsize=(10, 6))
plt.bar(range(len(importance_score)), importance_score)  # 使用 bar() 绘制柱状图
plt.title('Importance_Score')
plt.xlabel('Index')
plt.ylabel('Importance_Score')
plt.grid(True)
plt.savefig(f'../motif/data1/Importance_Score.svg')
plt.show()
plt.close()
# 取平均（如果需要）
avg_attr1 = abs(total_attr1) / len(test_dataset)
avg_attr2 = abs(total_attr2) / len(test_dataset)
avg_attr3 = abs(total_attr3) / len(test_dataset)
avg_attr4 = abs(total_attr4) / len(test_dataset)
avg_attr5 = abs(total_attr5) / len(test_dataset)
avg_attr6 = abs(total_attr6) / len(test_dataset)
avg_attr7 = abs(total_attr7) / len(test_dataset)
avg_attr8 = abs(total_attr8) / len(test_dataset)
avg_attr9 = abs(total_attr9) / len(test_dataset)
avg_attr10 = abs(total_attr10) / len(test_dataset)

from plot import plot_logo
# 创建一个包含 5 个子图的图形，设置每个图的大小
plt.figure(figsize=(40, 25))

custom_colors = {
    'A': 'red',
    'C': 'blue',
    'G': 'orange',
    'U': 'green'
}

# 绘制第一个图
ax1 = plt.subplot(10, 1, 1)  # 10 行 1 列，选择第 1 个位置
plot_logo(avg_attr1[0, :, :], ax=ax1,color=custom_colors)  # 绘制特征重要性 logo 图
ax1.set_title('Midpoint of [0-100]:index=50', fontsize=30)

# 绘制第二个图
ax2 = plt.subplot(10, 1, 2)  # 10 行 1 列，选择第 2 个位置
plot_logo(avg_attr2[0, :, :], ax=ax2,color=custom_colors)  # 绘制特征重要性 logo 图
ax2.set_title('Midpoint of [100-200]:index=150', fontsize=30)

# 绘制第三个图
ax3 = plt.subplot(10, 1, 3)  # 10 行 1 列，选择第 3 个位置
plot_logo(avg_attr3[0, :, :], ax=ax3,color=custom_colors)  # 绘制特征重要性 logo 图
ax3.set_title('Midpoint of [200-300]:index=250', fontsize=30)

# 绘制第四个图
ax4 = plt.subplot(10, 1, 4)  # 10 行 1 列，选择第 4 个位置
plot_logo(avg_attr4[0, :, :], ax=ax4,color=custom_colors)  # 绘制特征重要性 logo 图
ax4.set_title('Midpoint of [300-400]:index=350', fontsize=30)

# 绘制第五个图
ax5 = plt.subplot(10, 1, 5)  # 10 行 1 列，选择第 5 个位置
plot_logo(avg_attr5[0, :, :], ax=ax5,color=custom_colors)  # 绘制特征重要性 logo 图
ax5.set_title('Midpoint of [400-500]:index=450', fontsize=30)

# 绘制第六个图
ax6 = plt.subplot(10, 1, 6)  # 10 行 1 列，选择第 6 个位置
plot_logo(avg_attr6[0, :, :], ax=ax6,color=custom_colors)  # 绘制特征重要性 logo 图
ax6.set_title('Midpoint of [500-600]:index=550', fontsize=30)

# 绘制第七个图
ax7 = plt.subplot(10, 1, 7)  # 10 行 1 列，选择第 7 个位置
plot_logo(avg_attr7[0, :, :], ax=ax7,color=custom_colors)  # 绘制特征重要性 logo 图
ax7.set_title('Midpoint of [600-700]:index=650', fontsize=30)

# 绘制第八个图
ax8 = plt.subplot(10, 1, 8)  # 10 行 1 列，选择第 8 个位置
plot_logo(avg_attr8[0, :, :], ax=ax8,color=custom_colors)  # 绘制特征重要性 logo 图
ax8.set_title('Midpoint of [700-800]:index=750', fontsize=30)

# 绘制第九个图
ax9 = plt.subplot(10, 1, 9)  # 10 行 1 列，选择第 9 个位置
plot_logo(avg_attr9[0, :, :], ax=ax9,color=custom_colors)  # 绘制特征重要性 logo 图
ax9.set_title('Midpoint of [800-900]:index=850', fontsize=30)

# 绘制第十个图
ax10 = plt.subplot(10, 1, 10)  # 10 行 1 列，选择第 10 个位置
plot_logo(avg_attr10[0, :, :], ax=ax10,color=custom_colors)  # 绘制特征重要性 logo 图
ax10.set_title('Midpoint of [900-1000]:index=950', fontsize=30)

# 调整图间距，防止标题和图像重叠
plt.tight_layout()
# 保存图像
plt.savefig(f'../motif/data1/motifs_combined.svg')
# 显示图像
plt.show()
