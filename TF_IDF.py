import os
import itertools
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from torch.utils.data import DataLoader, TensorDataset
from models.feature_model import Lucky

# 设置 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_file(data_type, file_index):
    """读取数据文件"""
    datas_neg = pd.read_csv(f"data/other/{data_type}/{file_index}-0.csv")
    datas_pos = pd.read_csv(f"data/other/{data_type}/{file_index}-1.csv")
    seq = list(datas_neg['data']) + list(datas_pos['data'])
    label = list(datas_neg['label']) + list(datas_pos['label'])
    seq = [s.replace(' ', '').replace('U', 'T') for s in seq]
    return seq, label


def encode_sequence_1mer(sequences, max_seq):
    """1-mer 编码"""
    k = 1
    all_kmer = [''.join(p) for p in itertools.product(['A', 'T', 'C', 'G', '-'], repeat=k)]
    kmer_dict = {all_kmer[i]: i for i in range(len(all_kmer))}

    encoded_sequences = []
    max_length = max_seq if k == 1 else max_seq // k
    for seq in sequences:
        encoded_seq = []
        start_site = len(seq) // 2 - max_length // 2
        for i in range(start_site, start_site + max_length, k):
            encoded_seq.append(kmer_dict.get(seq[i:i + k], 0))
        encoded_sequences.append(encoded_seq + [0] * (max_length - len(encoded_seq)))
    return np.array(encoded_sequences)


def load_model_and_data(params):
    """加载模型和数据"""
    train_x, train_y = read_file(data_type='train', file_index=params['data_index'])
    train_x = encode_sequence_1mer(np.array(train_x), max_seq=params['seq_len'])

    train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)

    model = Lucky().to(device)
    model.load_state_dict(torch.load(f"save/mymodel/all/seed{params['seed']}.pth")['state_dict'])
    return model, train_loader


def compute_tf(loader, model, codebook_feature):
    """计算 TF 矩阵"""
    all_tf = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            outputs = model(x)
            quantized = outputs[5]  # 获取 VQ-VAE 量化特征
            quantized_indices = torch.argmin(torch.cdist(quantized, codebook_feature.unsqueeze(0)), dim=-1)

            batch_size = quantized.shape[0]
            tf_matrix = torch.zeros((batch_size, codebook_feature.shape[0]), dtype=torch.float32, device=device)
            for i in range(batch_size):
                indices, counts = torch.unique(quantized_indices[i], return_counts=True)
                tf_matrix[i, indices] = counts.float()
            all_tf.append(tf_matrix)
    return torch.cat(all_tf, dim=0)


def compute_idf(tf_matrix):
    """计算 IDF 向量"""
    N = 10  # 10 个数据集
    df = (tf_matrix > 0).sum(dim=1)  # 计算每个特征在多少个数据集中出现
    idf = torch.log(1 + N / df.float())
    return idf  # (512,)


import numpy as np

def reorder_features(tf_idf_matrix):
    """重新排序特征，使其更具生物学解释性"""
    tf_idf_matrix = tf_idf_matrix.cpu().numpy()  # 转换为 NumPy 数组

    # **步骤 1：确定每个特征的主要修饰类型**
    max_indices = np.argmax(tf_idf_matrix, axis=1)  # 每个特征（行）最相关的修饰类型

    # **步骤 2：按照主要修饰类型对特征进行分组**
    sorted_indices = np.argsort(max_indices)  # 先按照修饰类型进行排序
    sorted_tf_idf = tf_idf_matrix[sorted_indices, :]  # 先按修饰类型重新排列矩阵
    max_values_sorted = np.max(sorted_tf_idf, axis=1)  # 计算排序后的最高 TF-IDF 值

    # **步骤 3：在每种修饰类型内部，按照特异性分数排序**
    unique_modifications = np.unique(max_indices)
    final_sorted_indices = []

    for mod in unique_modifications:
        mod_indices = np.where(max_indices[sorted_indices] == mod)[0]  # 找出当前修饰类型的所有特征索引
        mod_sorted = mod_indices[np.argsort(-max_values_sorted[mod_indices])]  # 该修饰类型内按照特异性分数降序排序
        final_sorted_indices.extend(mod_sorted)

    # **最终的排序**
    sorted_tf_idf = sorted_tf_idf[final_sorted_indices, :]
    return sorted_tf_idf


def main():
    num_datasets = 10  # 10 个数据集
    feature_dim = 512  # 512 个特征

    loader_list = []
    tf_list = []

    for index in range(1, num_datasets + 1):
        params = {'batch_size': 32, 'seq_len': 501, 'data_index': index, 'seed': 0}
        model, train_loader = load_model_and_data(params)
        loader_list.append(train_loader)

        codebook_feature = model(torch.zeros(1, 501, dtype=torch.long).to(device))[6]

        tf_vector = compute_tf(train_loader, model, codebook_feature)
        tf_vector_sum = tf_vector.sum(dim=0, keepdim=True)
        tf_vector_mean = tf_vector_sum / tf_vector.shape[0]  # (1, 512)
        tf_list.append(tf_vector_mean)

    global_tf = torch.cat(tf_list, dim=0).T  # (10, 512) -> (512, 10)
    global_idf = compute_idf(global_tf)  # (512,)

    tf_idf_matrix = global_tf * global_idf.unsqueeze(1)  # (512, 10)

    # **归一化 (Z-score) 在排序前执行**
    df = pd.DataFrame(tf_idf_matrix.cpu().numpy(), columns=[f"Dataset_{i + 1}" for i in range(num_datasets)])
    df = df.dropna(how='all', subset=df.columns)
    df = df.apply(lambda row: (row - row.mean()) / row.std(), axis=1)  # Z-score归一化
    #df = df + abs(df.min().min())  # 将所有数据加上最小值的绝对值，确保数据全为正值
    tf_idf_matrix = torch.tensor(df.values, dtype=torch.float32)

    sorted_tf_idf = reorder_features(tf_idf_matrix)

    # **保存结果**
    df_sorted = pd.DataFrame(sorted_tf_idf, columns=[f"Dataset_{i + 1}" for i in range(num_datasets)])
    df_sorted.to_csv("tf_idf_matrix_sorted.csv", index=False)

    print("TF-IDF 矩阵计算完成 ✅，已归一化并按特异性分数排序")

    tf_idf_matrix = pd.read_csv('tf_idf_matrix_sorted.csv')

    plt.figure(figsize=(20, 60))

    colors = ['#acafcd', '#90a1ca', '#c3dcf0', '#d6b7d1', '#C37BA4', '#c6d9c1', '#98df8a', '#8FD0D9', '#F6D997',
              '#ffb07d']

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
    plt.show()
    plt.savefig('Z-score.png')
    plt.close()


if __name__ == '__main__':
    main()