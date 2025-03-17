import itertools
# from models.mymodel import Lucky
#from comparison.moss_m7g import Lucky
import os
import gc
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.metrics import davies_bouldin_score
import sys
sys.path.append('../')
from models.mymodel import Lucky
#from comparison.moss_m7g import Lucky
from comparison.train_other_methods import LSTM, CNN, Transformer, GRU
from matplotlib.colors import LinearSegmentedColormap
# ========== 全局配置 ==========
plt.rcParams['svg.fonttype'] = 'none'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    'batch_size': 128,
    'data_index': 6,  # 示例文件索引
    'seq_len': 501,  # 序列长度
    'seed': 3
}

# params = {
#             'kernel_num': 4096,
#             'topk': 128,
#             'lr': 0.0001,
#             'batch_size': 128,
#             'epoch': 300,
#             'seq_len': 501,
#             'saved_model_name': 'diff_len_',
#             'data_index': 3,
#             'seed': 0
#         }
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
    #model = Lucky(kernel_num=4096, topk=128).to(device)
    #model.load_state_dict(torch.load(f"../save/comparison/moss_m7g/data{params['data_index']}/seed{params['seed']}.pth")['state_dict'])
    model.load_state_dict(torch.load(f"../save/mymodel/data{params['data_index']}/seed{params['seed']}.pth")['state_dict'])

    return model, train_loader

def train_model_and_get_features(model, loader):
    """
    获取模型输出特征和标签。
    """
    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)  # 模型输出
            logits = outputs[-1]
            features.append(logits.cpu())  # 只对logits做cpu操作
            labels.append(y)

    return torch.cat(features), torch.cat(labels)

# ========== 可视化：使用t-SNE降维并可视化 ==========


def draw_tsne_circle(cmap='summer'):
    """
    使用 t-SNE 对特征进行降维并绘制散点图，计算Davies-Bouldin Index来评估降维效果。
    """
    plt.rcParams['svg.fonttype'] = 'none'
    gc.collect()

    # 加载模型和数据
    model, train_loader = load_model_and_data()

    # 获取特征和标签
    features, labels = train_model_and_get_features(model, train_loader)

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(features.numpy())

    # L2正则化：归一化操作
    normalizer = preprocessing.Normalizer(norm='l2').fit(X_2d)
    X_2d_normalized = normalizer.transform(X_2d)

    # 计算Davies-Bouldin Index（DBI）
    dbi_original = davies_bouldin_score(X_2d, labels.numpy())
    print(f"Davies-Bouldin Index (Original Vectors): {dbi_original}")

    #matplotlib.rcParams['svg.fonttype'] = 'none'
    # 绘制散点图
    plt.figure(figsize=(2, 2))
    plt.grid(False)


    # 绘制散点图
    plt.scatter(X_2d_normalized[:, 0], X_2d_normalized[:, 1], c=labels, cmap=cmap, alpha=0.1)
    # plt.colorbar()

    # 设置标题
    plt.title(f"VQ-MethyINet (DBI: {dbi_original:.2f})")
    #plt.title(f"CNN-Predictor (DBI: {dbi_original:.2f})")
    #plt.title(f"Moss-m7g (DBI: {dbi_original:.2f})")
    plt.tight_layout()
    # 保存为PDF
    #plt.savefig(f'tsne_{params["data_index"]}_circle.svg', format='svg')
    plt.savefig(f'../results/tense/data6/tsne_mymodel.svg', format='svg')
    plt.show()


colors = ['#003366', '#006400']   # 蓝色到绿色的渐变
n_bins = 100  # 色阶数量
cmap_name = 'blue_green_cmap'

# 创建 colormap
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
draw_tsne_circle(cm)
