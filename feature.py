import os
import itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.feature_model import Lucky

# 设置 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    test_x, test_y = read_file(data_type='test', file_index=params['data_index'])
    test_x = encode_sequence_1mer(np.array(test_x), max_seq=params['seq_len'])

    test_dataset = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    model = Lucky().to(device)
    model.load_state_dict(torch.load(f"save/mymodel/all/seed{params['seed']}.pth")['state_dict'])
    return model, test_loader


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
    return torch.cat(all_tf, dim=0)  # 对 batch 取平均，返回 (512,)


def compute_idf(tf_matrix):
    """计算 IDF 向量"""
    N = 10
    df = (tf_matrix > 0).sum(dim=1)  # 计算每个特征在多少个数据集中出现
    idf = torch.log(N / (1 + df.float()))
    return idf  # (512,)


def rearrange_matrix(tf_idf_matrix, feature_dim=512, num_datasets=10):
    """按照特异性分数重新排列特征谱矩阵"""
    # 计算每个特征的最高特异性分数，并找到对应的RNA修饰类型
    max_spec_scores, max_spec_types = tf_idf_matrix.max(dim=1)  # 计算每个特征的最高分数和对应的修饰类型

    # 创建特征索引的排序
    sorted_indices = torch.argsort(max_spec_scores, descending=True)

    # 重新排列矩阵的列
    rearranged_matrix = tf_idf_matrix[sorted_indices]

    # 对每个修饰类型内的特征按特异性分数从高到低排序
    rearranged_matrix_sorted_within_type = []
    for feature_type in range(num_datasets):
        type_indices = sorted_indices[max_spec_types == feature_type]  # 获取当前修饰类型的特征
        sorted_by_score = type_indices[torch.argsort(max_spec_scores[type_indices], descending=True)]  # 按分数排序
        rearranged_matrix_sorted_within_type.append(tf_idf_matrix[sorted_by_score])

    rearranged_matrix_sorted_within_type = torch.cat(rearranged_matrix_sorted_within_type, dim=0)

    return rearranged_matrix_sorted_within_type


def main():
    num_datasets = 10  # 10 个数据集
    feature_dim = 512  # 512 个特征

    loader_list = []
    tf_list = []

    for index in range(1, num_datasets + 1):
        params = {'batch_size': 32, 'seq_len': 501, 'data_index': index, 'seed': 0}
        model, test_loader = load_model_and_data(params)
        loader_list.append(test_loader)

        # 计算 codebook_feature
        codebook_feature = model(torch.zeros(1, 501, dtype=torch.long).to(device))[6]

        # 计算 TF
        tf_vector = compute_tf(test_loader, model, codebook_feature)
        tf_vector_sum = tf_vector.sum(dim=0, keepdim=True)
        tf_vector_mean = tf_vector_sum / tf_vector.shape[0]  # 形状变为 (1, 512)
        tf_list.append(tf_vector_mean)  # (1, 512)

    global_tf = torch.cat(tf_list, dim=0)  # (10, 512)
    global_tf = global_tf.T  # 变成 (10, 512)
    global_idf = compute_idf(global_tf)  # (512,)

    tf_idf_matrix = global_tf * global_idf.unsqueeze(1)  # (512, 10)

    # 重新排列 TF-IDF 矩阵
    rearranged_matrix = rearrange_matrix(tf_idf_matrix)

    # 保存结果
    df = pd.DataFrame(rearranged_matrix.cpu().numpy(), columns=[f"Dataset_{i + 1}" for i in range(num_datasets)])
    df.to_csv("rearranged_tf_idf_matrix.csv", index=False)
    print("TF-IDF 矩阵重新排列并保存完成 ✅")


if __name__ == '__main__':
    main()
