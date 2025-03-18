import os
import sys
sys.path.append('../')
import itertools
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from VQRNA_spectrum import VQRNA
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_fasta(file):
    seq = []
    label = []
    with open(file) as fasta:
        for line in fasta:
            line = line.replace('\n', '')
            if line.startswith('>'):
                # label.append(int(line[-1]))
                if 'neg' in line:
                    label.append(0)
                else:
                    label.append(1)
            else:
                seq.append(line.replace('U', 'T'))

    return seq, label

def read_file(data_type, file_index):

    datas_neg = f"../data/{data_type}/{file_index}-0.fasta"
    neg_seq, neg_label = read_fasta(datas_neg)
    datas_pos = f"../data/{data_type}/{file_index}-1.fasta"
    pos_seq, pos_label = read_fasta(datas_pos)
    seq = list(neg_seq) + list(pos_seq)
    label = list(neg_label) + list(pos_label)

    return seq, label


def encode_sequence_1mer(sequences, max_seq):
    """1-mer"""
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
    """Load the model and data"""
    train_x, train_y = read_file(data_type='train', file_index=params['data_index'])
    train_x = encode_sequence_1mer(np.array(train_x), max_seq=params['seq_len'])

    train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)

    model = VQRNA().to(device)
    model.load_state_dict(torch.load(f"../saved_models/feature_spectrum_seed0.pth")['state_dict'])
    return model, train_loader


def compute_tf(loader, model, codebook_feature):
    """Compute the TF matrix"""
    all_tf = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            outputs = model(x)
            quantized = outputs[5]  # Obtain VQ-VAE quantized features
            quantized_indices = torch.argmin(torch.cdist(quantized, codebook_feature.unsqueeze(0)), dim=-1)

            batch_size = quantized.shape[0]
            tf_matrix = torch.zeros((batch_size, codebook_feature.shape[0]), dtype=torch.float32, device=device)
            for i in range(batch_size):
                indices, counts = torch.unique(quantized_indices[i], return_counts=True)
                tf_matrix[i, indices] = counts.float()
            all_tf.append(tf_matrix)
    return torch.cat(all_tf, dim=0)


def compute_idf(tf_matrix):
    """Compute the IDF vector"""
    N = 10  # 10 datasets
    df = (tf_matrix > 0).sum(dim=1)  # Calculate the number of datasets in which each feature appears
    idf = torch.log(1 + N / df.float())
    return idf  # (512,)


def reorder_features(tf_idf_matrix):
    """Reorder features to enhance biological interpretability."""
    tf_idf_matrix = tf_idf_matrix.cpu().numpy()  # Convert to NumPy array

    # **Step 1: Determine the primary modification type for each feature**
    max_indices = np.argmax(tf_idf_matrix, axis=1)  # Most relevant modification type for each feature (row)

    # **Step 2: Group features by their primary modification type**
    sorted_indices = np.argsort(max_indices)  # Sort by modification type
    sorted_tf_idf = tf_idf_matrix[sorted_indices, :]  # Rearrange matrix by modification type
    max_values_sorted = np.max(sorted_tf_idf, axis=1)  # Compute the highest TF-IDF value after sorting

    # **Step 3: Within each modification type, sort by specificity score**
    unique_modifications = np.unique(max_indices)
    final_sorted_indices = []

    for mod in unique_modifications:
        mod_indices = np.where(max_indices[sorted_indices] == mod)[0]  # Find feature indices for the current modification type
        mod_sorted = mod_indices[np.argsort(-max_values_sorted[mod_indices])]  # Sort in descending order of specificity score
        final_sorted_indices.extend(mod_sorted)

    # **Final ordering**
    sorted_tf_idf = sorted_tf_idf[final_sorted_indices, :]
    return sorted_tf_idf


def main():
    num_datasets = 10  # 10 datasets
    feature_dim = 512  # 512 features

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

    df = pd.DataFrame(tf_idf_matrix.cpu().numpy(), columns=[f"Dataset_{i + 1}" for i in range(num_datasets)])
    df = df.dropna(how='all', subset=df.columns)
    df = df.apply(lambda row: (row - row.mean()) / row.std(), axis=1)  # Z-score normalization
    #df = df + abs(df.min().min())
    tf_idf_matrix = torch.tensor(df.values, dtype=torch.float32)

    sorted_tf_idf = reorder_features(tf_idf_matrix)

    # **save TF-IDF matrix**
    df_sorted = pd.DataFrame(sorted_tf_idf, columns=[f"Dataset_{i + 1}" for i in range(num_datasets)])
    df_sorted.to_csv("tf_idf_matrix_sorted.csv", index=False)

    print("TF-IDF matrix computation completed✅, and sorted by specificity score.")

    tf_idf_matrix = pd.read_csv('tf_idf_matrix_sorted.csv')

    plt.figure(figsize=(20, 60))

    colors = ['#acafcd', '#90a1ca', '#c3dcf0', '#d6b7d1', '#C37BA4', '#c6d9c1', '#98df8a', '#8FD0D9', '#F6D997',
              '#ffb07d']

    for i in range(10):
        plt.subplot(10, 1, i + 1)
        values = tf_idf_matrix.iloc[:, i]
        plt.bar(range(len(values)), values, color=colors[i], label=f'Dataset_{i + 1}')

        plt.title('')
        plt.xlabel('')
        plt.ylabel('')
        plt.legend().set_visible(False)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('VQRNA_feature_spectrum.svg', format='svg')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()