#!/usr/bin/env Python
# coding=utf-8

from collections import Counter
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
import itertools
import math
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import pandas as pd
import random

import time
import torch
tqdm.pandas(ascii=True)
import os
from argparse import ArgumentParser
from functools import reduce

from itertools import product
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from termcolor import colored
# from models.capsulnet import Capsulnet, MarginLoss
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from models.mymodel import Lucky
#from length.mymodel_1001 import Lucky
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda", 0)

params = {
        'lr': 0.001,
        'batch_size': 64,
        'epoch': 300,
        'seq_len': 501,
        'saved_model_name': 'diff_len_',
        'seed': 45,
        'data_index': 2,
        'patience': 30,
        'loss1': 0.5643794278368945,
        'loss2': 0.0688552625562815,
        'loss3': 0.4506530513121981
    }


class AP(nn.Module):
    def __init__(self, d_hidden):
        super(AP, self).__init__()
        self.linear = nn.Linear(d_hidden, d_hidden)
        self.balance = nn.Linear(d_hidden, 1)
        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.linear.in_features +
                                  self.linear.out_features)
        self.linear.weight.data.uniform_(-r, r)
        self.linear.bias.data.fill_(0)

    def forward(self, features):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        max_len = features.size(1)
        lengths = torch.ones(features.size(0), device=features.device) * max_len
        mask = torch.arange(max_len).expand(features.size(0), features.size(1)).to(features.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        mask_features = features.masked_fill(mask == 0, -1000)
        mask_features = mask_features.sort(dim=1, descending=True)[0]

        # embedding-level
        embed_weights = F.softmax(mask_features, dim=1)
        embed_features = (mask_features * embed_weights).sum(1)

        # token-level
        # token_weights = [B x K x D]
        mask_features = mask_features.masked_fill(mask == 0, 0)
        token_weights = self.linear(mask_features)
        token_weights = F.softmax(self.relu(token_weights),
                                  dim=1)
        token_features = (mask_features * token_weights).sum(dim=1)
        fusion_features = torch.cat([token_features.unsqueeze(1),
                                     embed_features.unsqueeze(1)],
                                    dim=1)
        fusion_weights = F.softmax(self.balance(fusion_features),
                                   dim=1)
        pool_features = (fusion_features * fusion_weights).sum(1)

        # return pool_features, fusion_weights.squeeze()
        return pool_features

def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap


def draw_heatmap(score, name, attention_type):
    matplotlib.rcParams['svg.fonttype'] = 'none'

    plt.figure(figsize=(6, 5))
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['figure.dpi'] = 200
    # plot
    sns.set()
    # scores = (scores - scores.min()) / ( scores.max() - scores.min())
    # ax = sns.heatmap(scores, cmap='Greens')
    ax1 = sns.heatmap(score, cmap='YlGnBu')
    if attention_type == 'all':
        plt.savefig(f'attention/all_layer&head_attentions/data2/all_laye1024.svg')
    elif attention_type == 'last':
        plt.savefig(f'attention/last_layer&first_head_attentions/data2/111last_layer{name}.svg')
    elif attention_type == 'first':
        plt.savefig(f'attention/first_layer&first_head_attentions/data2/first_layer1024.svg')

    # plt.show()
    plt.clf()
# torch version

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
    """
    读取数据文件并提取序列和标签。
    """
    datas_neg = pd.read_csv(f"data/other/{data_type}/{file_index}-0.csv")
    datas_pos = pd.read_csv(f"data/other/{data_type}/{file_index}-1.csv")
    seq = list(datas_neg['data']) + list(datas_pos['data'])
    label = list(datas_neg['label']) + list(datas_pos['label'])

    seq = [s.replace(' ', '').replace('U', 'T') for s in seq]  # 序列标准化
    return seq, label

def encode_sequence_1mer(sequences, max_seq):
    k = 1
    overlap = False

    # 生成所有可能的k-mer字典，包含 'A', 'T', 'C', 'G', '-'
    all_kmer = [''.join(p) for p in itertools.product(['A', 'T', 'C', 'G', '-'], repeat=k)]
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
            kmer = seq[i:i + k]
            # 跳过包含 'N' 的k-mer
            if 'N' in kmer:
                continue  # 跳过含 'N' 的片段
            encoded_seq.append(kmer_dict[kmer])

        # 填充到 max_length
        encoded_sequences.append(encoded_seq + [0] * (max_length - len(encoded_seq)))

    return np.array(encoded_sequences)

def to_log(log):
    with open(f"results/train_result.log", "a+") as f:
        f.write(log + '\n')

# ========================================================================================



def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # Accuracy
    ACC = float(tp + tn) / test_num

    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)

    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    if (tp + fp) == 0:
        PRE = 0
    else:
        PRE = float(tp) / (tp + fp)

    BACC = 0.5 * Sensitivity + 0.5 * Specificity

    performance = [ACC, BACC, Sensitivity, Specificity, MCC, AUC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data

def evaluate(data_iter, net):
    pred_prob = []
    label_pred = []
    label_real = []

    for j, (data, labels) in enumerate(data_iter, 0):
        labels = labels.to(device)
        data = data.to(device)
        output, _, vq_loss, data_recon, perplexity = net(data)

        outputs_cpu = output.cpu()
        y_cpu = labels.cpu()
        pred_prob_positive = outputs_cpu[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + output.argmax(dim=1).tolist()
        label_real = label_real + y_cpu.tolist()
    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real)
    return performance, roc_data, prc_data

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)

def get_attention(representation, start=0, end=0):
    attention = representation[-1]
    # print(attention)
    # print(len(attention))
    # print(attention[0].shape)
    # print(attention[0].squeeze(0).shape) torch.Size([12, 43, 43])

    """
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    attn = format_attention(attention)
    # print(attn.shape) torch.Size([12, 12, 43, 43])

    attn_score = []
    # attn_score=torch.sum(attn, dim=1).squeeze()
    for i in range(attn.shape[3]):
        # only use cls token, because use pool out
        attn_score.append(float(attn[start:end + 1, :, 0, i].sum()))

    # print(len(attn_score)) 41
    return attn_score

def full_attention(representation, attention_type):
    # if attention_type == 'last':
    #     attention = representation[-1]
    #     attn = format_attention(attention)
    #     attn_score = torch.sum(attn, dim=1).squeeze()  # last layer & all head

    if attention_type == 'all':
        attention = representation
        attn = format_attention(attention)
        attn_score = torch.sum(torch.sum(attn, dim=0).squeeze(), dim=0).squeeze()  # all layer & all head

    # print(len(attn_score)) 41
    return attn_score

def visualize_one_sequence(attention_1024,attention_type):

    # if attention_type == 'last':
    #     attention_1024 = get_attention(attention_1024, -1, 0)  #last layer & first head
    if attention_type == 'first':
        attention_1024 = get_attention(attention_1024, 0, 0)  #first layer & first head

    attention_scores_1024 = np.array(attention_1024).reshape(np.array(attention_1024).shape[0], 1)
    scores_1024 = attention_scores_1024.reshape(1, attention_scores_1024.shape[0])

    return scores_1024

def my_evaluation_method(params):

#################################################### prepare data #####################################################
    # get label
    train_x, train_y = read_file(data_type='train', file_index=params['data_index'])
    valid_x, valid_y = read_file(data_type='valid', file_index=params['data_index'])
    test_x, test_y = read_file(data_type='test', file_index=params['data_index'])

    train_x, train_y = np.array(train_x), np.array(train_y)
    valid_x, valid_y = np.array(valid_x), np.array(valid_y)
    test_x, test_y = np.array(test_x), np.array(test_y)

    test_x = encode_sequence_1mer(test_x, max_seq=params['seq_len'])
    train_x = encode_sequence_1mer(train_x, max_seq=params['seq_len'])
    valid_x = encode_sequence_1mer(valid_x, max_seq=params['seq_len'])

    train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    valid_dataset = TensorDataset(torch.tensor(valid_x), torch.tensor(valid_y))
    test_dataset = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    # valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

################################################### prepare data ######################################################


################################################### attention map #####################################################
# 加载模型
    model = Lucky().to(device)
    model.load_state_dict(torch.load(f"save/vq_size/1024/data{params['data_index']}/seed{params['seed']}.pth")['state_dict'])
    #model.load_state_dict(torch.load(f"save/mymodel/data{params['data_index']}/seed{params['seed']}.pth")['state_dict'])

    model.eval()
    model = model.to(device)
    test_performance, test_roc_data, test_prc_data = evaluate(test_loader, model)
    print(test_performance)

    scores_1024_avg = []

    #attention_type = 'first'
    attention_type = 'all'

    #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    for seq, label in tqdm(train_loader):
        seq, label = seq.to(device), label.to(device)
        # train_attention = train_attention.to(device)
        logits, attention_1024, vq_loss, data_recon, perplexity = model(seq)
        # print(attention_1024[0].shape)
        #scores_1024 = visualize_one_sequence(attention_1024,attention_type)
        scores_1024 = np.array(full_attention(attention_1024, attention_type).cpu().detach())

        scores_1024_avg.append(scores_1024)


    scores_1024_avg = np.sum(np.stack(scores_1024_avg, axis=0), axis=0)
    scores_1024_avg = minmax_scale(scores_1024_avg, axis=1)


    seq_len = params['seq_len']
    draw_heatmap(scores_1024_avg, seq_len, attention_type)


################################################### attention map #####################################################

def main():

    seed = params['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    my_evaluation_method(params)

if __name__ == '__main__':
    main()