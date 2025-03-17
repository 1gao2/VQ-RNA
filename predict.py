import os

import torch
import numpy as np
import itertools
from length.mymodel_101 import Lucky

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = {
    'seq_len': 101,
    'data_index': 2,
    'seed': 2
}


# 1-mer 编码函数
def encode_sequence_1mer(sequence, max_seq):
    k = 1  # k-mer大小
    all_kmer = [''.join(p) for p in itertools.product(['A', 'T', 'C', 'G', '-'], repeat=k)]
    kmer_dict = {all_kmer[i]: i for i in range(len(all_kmer))}

    encoded_seq = []
    max_length = max_seq if k == 1 else max_seq // k
    start_site = max(0, len(sequence) // 2 - max_length // 2)
    for i in range(start_site, start_site + max_length, k):
        encoded_seq.append(kmer_dict.get(sequence[i:i + k], 0))  # 默认不存在时为0

    return np.array(encoded_seq + [0] * (max_length - len(encoded_seq)))


# 加载模型
def load_model():
    model = Lucky().to(device)
    model.load_state_dict(
        torch.load(f"save/length/length101/data{params['data_index']}/seed{params['seed']}.pth")['state_dict'])
    model.eval()
    return model


# 预测函数
def predict_sequence(model, sequence):
    sequence = sequence.replace(' ', '').replace('U', 'T')  # 预处理
    encoded_seq = encode_sequence_1mer(sequence, params['seq_len'])
    input_tensor = torch.tensor(encoded_seq).unsqueeze(0).to(device)  # 添加 batch 维度

    with torch.no_grad():
        output = model(input_tensor)  # 模型返回的是一个 tuple
        logits = output[0].cpu().numpy()  # 获取 logits
    prediction = np.argmax(logits)
    return prediction


if __name__ == '__main__':
    model = load_model()
    test_sequence = "GCACGCTGCAGCCCGGAGTCCCCGTTCACACTGAGGAACGGAGACCTGTGACCACAGCAGGCTGACAGATGGACAGAATCTCCCGTAGAAAGGTTTGGTTT"  # 示例序列
    pred_result = predict_sequence(model, test_sequence)
    print("Prediction:", pred_result)
