# 必要的模块导入
import os
import random
import itertools
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append('../')
from models.mymodel import Lucky
import umap
import matplotlib.pyplot as plt

tqdm.pandas(ascii=True)

# 设置全局变量
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = {
    'batch_size': 32,
    'data_index': 7,  # 示例文件索引
    'seq_len': 501,    # 序列长度
    'seed': 3
}

# ========== 数据处理 ==========

def read_file(data_type, file_index):
    """
    读取数据文件并提取序列和标签。
    """
    datas_neg = pd.read_csv(f"../data/other/{data_type}/{file_index}-0.csv")
    datas_pos = pd.read_csv(f"../data/other/{data_type}/{file_index}-1.csv")
    seq = list(datas_neg['data']) + list(datas_pos['data'])
    label = list(datas_neg['label']) + list(datas_pos['label'])

    seq = [s.replace(' ', '').replace('U', 'T') for s in seq]  # 序列标准化
    return seq, label


def encode_sequence_1mer(sequences, max_seq):
    """
    对序列进行1-mer编码。
    """
    k = 1  # k-mer大小
    all_kmer = [''.join(p) for p in itertools.product(['A', 'T', 'C', 'G', '-'], repeat=k)]
    kmer_dict = {all_kmer[i]: i for i in range(len(all_kmer))}

    encoded_sequences = []
    max_length = max_seq if k == 1 else max_seq // k

    for seq in sequences:
        encoded_seq = []
        start_site = len(seq) // 2 - max_length // 2
        for i in range(start_site, start_site + max_length, k):
            encoded_seq.append(kmer_dict.get(seq[i:i + k], 0))  # 默认不存在时为0
        encoded_sequences.append(encoded_seq + [0] * (max_length - len(encoded_seq)))

    return np.array(encoded_sequences)


# ========== 模型加载和测试 ==========

def load_model_and_data():
    """
    加载模型和验证数据。
    """
    # 加载数据
    train_x, train_y = read_file(data_type='train', file_index=params['data_index'])
    train_x, train_y = np.array(train_x), np.array(train_y)
    train_x = encode_sequence_1mer(train_x, max_seq=params['seq_len'])

    train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)

    # 加载模型
    model = Lucky().to(device)
    model.load_state_dict(torch.load(f"../save/mymodel/data{params['data_index']}/seed{params['seed']}.pth")['state_dict'])

    return model, train_loader


# ========== 可视化 ==========

def train_all_so_far(model, train_loader):
    """
    示例测试函数，返回特征和标签。
    """
    maps, vlabels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            outputs = model(x)  # 模型返回的是一个tuple
            logits = outputs[-1]

            maps.append(logits.cpu())  # 只对logits做cpu操作
            vlabels.append(y)

    maps = torch.cat(maps)
    vlabels = torch.cat(vlabels)
    return maps, vlabels



def draw_umap(x, vlabels):
    """
    使用UMAP进行降维并可视化。
    """
    x = x.cpu().numpy()
    umap_model = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_model.fit_transform(x)

    # 绘图
    plt.figure(figsize=(10, 8))
    colors = ['#C25759', '#D9B9D4']
    #colors = ['#41659e', '#88d7d2']  # 为每个类指定颜色
    #colors = ['#41659e', '#BA3E45']

    cmap = ListedColormap(colors)  # 创建颜色映射
    boundaries = [0, 1, 2]  # 定义边界
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=vlabels, cmap=cmap, s=50, alpha=0.8, norm=norm)
    # 添加颜色条（color bar）并设置ticks
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    # cbar.set_label('Classes', fontsize=14)
    cbar.ax.set_yticklabels(['0', '1'])  # 设置每个类的标签

    # 将标签放置在对应tick的中间
    tick_positions = np.arange(0.5, 2, 1)  # 在0.5到3.5之间生成ticks
    cbar.ax.set_yticks(tick_positions)  # 设置新的y ticks
    cbar.ax.set_yticklabels(['0', '1'], fontsize=20)  # 中间对齐标签

    plt.title(f'VQ-MethylNet', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    os.makedirs('../results/umap', exist_ok=True)
    index = {params['data_index']}
    plt.savefig(f'../results/umap/data{index}_VQ-MethylNet_UMAP.svg', format='svg')
    plt.show()


# ========== 主函数 ==========

def main():
    model, train_loader = load_model_and_data()
    maps, vlabels = train_all_so_far(model, train_loader)

    # # 筛选非零标签
    # nonZero = torch.abs(vlabels) > 0
    # maps = maps[nonZero]
    # vlabels = vlabels[nonZero]

    # 调用UMAP绘图
    draw_umap(maps, vlabels)


if __name__ == '__main__':
    main()
