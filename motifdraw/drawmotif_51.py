
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
from motifmodel.motifmodel_51 import Lucky
from tangermeme.plot import plot_logo
from tangermeme.ism import saturation_mutagenesis
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    'batch_size': 32,
    'data_index': 1,  # 示例文件索引
    'seq_len': 51,    # 序列长度
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
    model.load_state_dict(torch.load(f"../save/length/length51/data{params['data_index']}/seed{params['seed']}.pth")['state_dict'])

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

# 遍历 test_dataset 的所有样本
for i in range(len(test_dataset)):
    # 获取第 i 个样本的特征
    x = test_dataset[i][0].unsqueeze(dim=0)  # 取第 i 个样本并扩展维度
    x = F.one_hot(x, num_classes=4).transpose(1, 2).float()  # 将序列转为 one-hot 编码

    # 计算特征重要性
    X_attr1 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=25 - 25, end=25 + 25, device='cpu')
    X_attr2 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=0, end=50, device='cpu')

    # 累加特征重要性
    if total_attr1 is None:
        total_attr1 = X_attr1  # 初始化累加矩阵
    else:
        total_attr1 += X_attr1  # 累加特征重要性

    if total_attr2 is None:
        total_attr2 = X_attr2  # 初始化累加矩阵
    else:
        total_attr2 += X_attr2  # 累加特征重要性

# Feature importance score
avg_attr2 = abs(total_attr2) / len(test_dataset)
importance_score = torch.sum(avg_attr2, dim=1)
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
ax1 = plt.subplot(1, 1, 1)  # 1 行 1 列，选择第 1 个位置
plot_logo(avg_attr1[0, :, :], ax=ax1,color=custom_colors)  # 绘制特征重要性 logo 图
ax1.set_title('Midpoint of the entire sequence:index=25', fontsize=60)

# 调整图间距，防止标题和图像重叠
plt.tight_layout()
# 保存图像
plt.savefig(f'../motif/data1/motifs_combined.svg')
# 显示图像
plt.show()
