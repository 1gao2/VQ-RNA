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
import logomaker
import seaborn
from matplotlib import pyplot as plt

seaborn.set_style("white")
from itertools import product
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from termcolor import colored
# from models.capsulnet import Capsulnet, MarginLoss
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from models.model import Lucky
from collections import Counter, defaultdict
import glob
from sklearn.preprocessing import MinMaxScaler

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda", 0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
        """
        Args:
          d_model:      dimension of embeddings
          dropout:      randomly zeroes-out some of the input
          max_length:   max sequence length
        """
        # inherit from Module
        super().__init__()

        # initialize dropout
        self.dropout = nn.Dropout(p=dropout)

        # create tensor of 0s
        pe = torch.zeros(max_length, d_model)

        # create position column
        k = torch.arange(0, max_length).unsqueeze(1)

        # calc divisor for positional encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        # calc sine on even indices
        pe[:, 0::2] = torch.sin(k * div_term)

        # calc cosine on odd indices
        pe[:, 1::2] = torch.cos(k * div_term)

        # add dimension
        pe = pe.unsqueeze(0)

        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer("pe", pe)

    def forward(self, x):

        # add positional encoding to the embeddings
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        # perform dropout
        return self.dropout(x)

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


def draw_heatmap(score, name):
    plt.figure(figsize=(10, 4))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    # plot
    sns.set()
    # scores = (scores - scores.min()) / ( scores.max() - scores.min())
    # ax = sns.heatmap(scores, cmap='Greens')
    ax1 = sns.heatmap(score, cmap='YlGnBu')
    plt.savefig(f'figures/{name}.svg')
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

def encode_sequence_1mer(sequences, max_seq):
    k = 1
    overlap = False

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
            encoded_seq.append(kmer_dict[seq[i:i+k]])

        encoded_sequences.append(encoded_seq+[0]*(max_length-len(encoded_seq)))

    return np.array(encoded_sequences)

def to_log(log):
    with open(f"results/train_result.log", "a+") as f:
        f.write(log + '\n')

# ========================================================================================

def rna_encoder(train_loader, valid_loader, test_loader, lr_r, epoch_r, batch_size_r, saved_model_name, train_attention=None):
    # Define model
    model = Lucky().to(device)
    # model = Model().to(device)

    # Optimizer and loss
    # opt = optim.AdamW(model.parameters(), lr=lr_r)
    opt = optim.Adam(model.parameters(), lr=lr_r, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    criterion_CE = nn.CrossEntropyLoss()
    best_acc = 0
    # Training loop
    early_stop = 0
    patience = 10
    for epoch in range(epoch_r):
        model.train()
        loss_ls = []
        t0 = time.time()
        for seq, label in tqdm(train_loader):
            seq, label = seq.to(device), label.to(device)
            # train_attention = train_attention.to(device)
            # output_feature, out_seq, logits = model(feature, seq)
            # print(seq.shape)
            logits, _ = model(seq, 4)
            # a = criterion_MA(logits_1, label)
            b = criterion_CE(logits, label)
            # b = criterion_CE(output_feature, label)
            # loss = a + b
            loss = b

            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_ls.append(loss.item())
        # Validation step (if needed)
        model.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_prc_data = evaluate(train_loader, model)
            valid_performance, valid_roc_data, valid_prc_data = evaluate(valid_loader, model)

        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'Train: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 16 + ' Valid Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                   + '\n[ACC, \tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            valid_performance[0], valid_performance[1], valid_performance[2], valid_performance[3],
            valid_performance[4], valid_performance[5]) + '\n' + '=' * 60
        valid_acc = valid_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]

        if valid_acc > best_acc:
            best_acc = valid_acc
            test_performance, test_roc_data, test_prc_data = evaluate(valid_loader, model)
            test_results = '\n' + '=' * 16 + colored(' Test Performance. Epoch[{}] ', 'red').format(
                epoch + 1) + '=' * 16 \
                           + '\n[ACC,\tBACC, \tSE,\t\tSP,\t\tAUC,\tPRE]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                test_performance[0], test_performance[1], test_performance[2], test_performance[3],
                test_performance[4], test_performance[5]) + '\n' + '=' * 60
            print(test_results)
            torch.save(model.state_dict(), f"saved_models/Lucky{test_performance[0]}.pth")
        else:
            early_stop += 1
            if early_stop > patience:
                print(f'early stop! best_acc: {best_acc}')
                break
    # Save model
    # torch.save(model.state_dict(), f"{save_path}/encoder_rna_1.pth")

    return best_acc

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
        output, _, _= net(data)

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

def get_attention(representation, start, end):
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
    # for i in range(1, attn.shape[3] - 1):
    for i in range(attn.shape[3]):
        # only use cls token, because use pool out
        attn_score.append(float(attn[start:end + 1, :, 0, i].sum()))

    # print(len(attn_score)) 41
    return attn_score
def visualize_one_sequence(attention_1024):

    # print(get_attention(attention_510, 1, 1))

    # attention_510 = get_attention(attention_510, 0, 0)
    attention_1024 = get_attention(attention_1024, 0, 0)

    # attention_scores_510 = np.array(attention_510).reshape(np.array(attention_510).shape[0], 1)
    attention_scores_1024 = np.array(attention_1024).reshape(np.array(attention_1024).shape[0], 1)

    # scores_510 = attention_scores_510.reshape(1, attention_scores_510.shape[0])
    scores_1024 = attention_scores_1024.reshape(1, attention_scores_1024.shape[0])

    # print(scores_510.shape)
    # print(scores_1024.shape)

    return scores_1024

def plotfun(motifpwm, title=None, ax=None, ylabel=True):

    color_scheme = {
        'A' : [0, 0.702, 0.345],
        'C' : [0.051, 0.333, 0.651],
        'G' : [1,0.604,0],
        'U' : [1,0.239,0],
    }

    motifpwm = pd.DataFrame(motifpwm,columns=['A','U','C','G'])
    crp_logo = logomaker.Logo(motifpwm,
                              shade_below=.8, # 灰色蒙版
                              fade_below=.8, # 透明度
                              # font_name='Arial Rounded MT Bold',
                              font_name='Arial',
                              color_scheme=color_scheme,
                             ax=ax, width=0.8)

    # style using Logo methods
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    crp_logo.style_xticks(rotation=90, fmt='%d', anchor=0)
    # if title is not None:
    #     crp_logo.ax.set_title(title,x=1.2, y=0.5)

    # style using Axes methods
    if ylabel:
        crp_logo.ax.set_ylabel("Motif score", labelpad=-1)
    crp_logo.ax.xaxis.set_ticks_position('none')
    crp_logo.ax.yaxis.set_ticks_position('left')

    crp_logo.ax.set_xticks([])
    crp_logo.ax.tick_params(axis='both', which='major', pad=-3)
    return crp_logo

def my_evaluation_method(params):

#################################################### prepare data ######################################################
    # get label

    train_x, train_y = read_fasta('data/train.fasta')
    valid_x, valid_y = read_fasta('data/valid.fasta')
    test_x, test_y = read_fasta('data/test.fasta')


    seq_len = params['seq_len']
    train_x, train_y = np.array(train_x), np.array(train_y)
    valid_x, valid_y = np.array(valid_x), np.array(valid_y)
    test_x, test_y = np.array(test_x), np.array(test_y)

    train_x = encode_sequence_1mer(train_x, max_seq=seq_len)
    valid_x = encode_sequence_1mer(valid_x, max_seq=seq_len)
    test_x = encode_sequence_1mer(test_x, max_seq=seq_len)

    train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    valid_dataset = TensorDataset(torch.tensor(valid_x), torch.tensor(valid_y))
    test_dataset = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    #
    # ################################################### attention map #####################################################
    # import seaborn as sns
    # model = Lucky(kernel_num=params['kernel_num'], topk=params['topk'])
    # # best_acc = 0.8961272891023716
    # # load_params(model, config.path_params)
    # model.load_state_dict(torch.load(f'saved_models/17_501_4096_128_0.8436645396536008.pth'))
    # model.eval()
    # model = model.to(device)
    # #
    # # # scores_1024_avg = np.zeros((1, 68))
    # #
    # test_performance, test_roc_data, test_prc_data = evaluate(valid_loader, model)
    # print(test_performance)
    # #
    # test_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    #
    # ################################################# top k words frequency ############################################
    # words = []            # 词表
    # probilities = []          # softmax后的概率值
    # for seq, label in tqdm(test_loader):
    #     seq, label = seq.to(device), label.to(device)
    #     # train_attention = train_attention.to(device)
    #     logits, word, probility = model(seq)
    #     words.append(word.cpu().detach())
    #     probilities.append(probility.cpu().detach())
    # #
    # pro_counts = defaultdict(float)      # 词表对应的总概率值字典
    # words = np.array(torch.cat(words, dim=0))
    # probilities = np.array(torch.cat(probilities, dim=0))
    #
    words = np.load(f'figures/temp_data/words_probilities/{seq_len}/words.npy')
    probilities = np.load(f'figures/temp_data/words_probilities/{seq_len}/words_probilities.npy')

    words = words.reshape(words.shape[1], -1)
    probilities = probilities.reshape(probilities.shape[1], -1)

    words_mid = words[words.shape[0]//2]
    probilities_mid = probilities[probilities.shape[0]//2]
    words_left = words[words.shape[0] // 2 - 1]
    probilities_left = probilities[probilities.shape[0] // 2 - 1]
    words_right = words[words.shape[0] // 2 + 1]
    probilities_right = probilities[probilities.shape[0] // 2 + 1]


    def count_num(words, probilities):
        count = Counter(words)
        unweighted_top_10 = count.most_common(10)
        unweighted_top_10_index = [item[0] for item in unweighted_top_10]

        weighted_dict = defaultdict(float)
        for i in range(words.shape[0]):
            weighted_dict[int(words[i])] += probilities[i]
        sortd_weighted_dict = sorted(weighted_dict.items(), key=lambda x: x[1])
        weighted_top_10_index = [item[0] for item in sortd_weighted_dict]
        weighted_top_10_index.reverse()

        return unweighted_top_10_index[:10], weighted_top_10_index[:10]

    unweighted_top_10_index_mid, weighted_top_10_index_mid = count_num(words_mid, probilities_mid)
    unweighted_top_10_index_left, weighted_top_10_index_left = count_num(words_left, probilities_left)
    unweighted_top_10_index_right, weighted_top_10_index_right = count_num(words_right, probilities_right)

    draw_motif(model, unweighted_top_10_index_mid, name='unweighted_top_10_index_mid')
    draw_motif(model, weighted_top_10_index_mid, name='weighted_top_10_index_mid')
    draw_motif(model, unweighted_top_10_index_left, name='unweighted_top_10_index_left')
    draw_motif(model, weighted_top_10_index_left, name='weighted_top_10_index_left')
    draw_motif(model, unweighted_top_10_index_right, name='unweighted_top_10_index_right')
    draw_motif(model, weighted_top_10_index_right, name='weighted_top_10_index_right')


    # return

    # words = np.load(f'figures/temp_data/words_probilities/{seq_len}/words.npy')
    # probilities = np.load(f'figures/temp_data/words_probilities/{seq_len}/words_probilities.npy')

#
# # 加权：计算每个词出现的频率，得到 pro_counts
#     for item in range(words.shape[0]):
#         for i in range(words.shape[1]):
#             for j in range(words.shape[2]):
#                 pro_counts[int(words[item][i][j])] += float(probilities[item][i][j])
#                     pro_counts[int(words[item][i][j])] = float(probilities[item][i][j])
#
#     # 储存 pro_counts 字典，大小为：（2, 词表大小），第一个值为key值，第二个为value值
#     pro_counts_to_saved = np.array([list(pro_counts.keys()), list(pro_counts.values())])
#     np.save(f'figures/temp_data/words_probilities/{seq_len}/pro_counts.npy', pro_counts_to_saved)
#     return


    # 计算词出现频率排名
    # pro_counts_data = np.load(f'figures/temp_data/words_probilities/{seq_len}/position_motif_counts.npy')   # 每个词出现的频率表
    # return
    # all_dict = {}
    # for i in range(pro_counts_data.shape[1]):     # 将numpy文件映射为dict
    #     all_dict[int(pro_counts_data[0, i])] = pro_counts_data[1, i]
    #
    # sorted_dict = sorted(all_dict.items(), key=lambda x: x[1])    # 按照频率大小进行排序
    #
    # 取出按频率排序后的词表
    # frequecy_word_rank = [item[0] for item in sorted_dict]
    # frequecy_word_rank.reverse()
    #
    #
    # # top_10 = [3444, 410, 3968, 2479, 220, 3524, 2157, 2812, 3435, 3502]
    #
    # # # 计算前十个词在每个位置的出现频率
    # top_10 = frequecy_word_rank[:10]
    # position_frequency_neg = np.zeros((10, words.shape[1]))
    # for l in range(words.shape[1]):  # bs, lenth, topk
    #     for item in range(words.shape[0]//2, words.shape[0]):
    #         for k in range(words.shape[2]):
    #             if words[item][l][k] in top_10:
    #                 position_frequency_neg[top_10.index(words[item][l][k])][l] += probilities[item][l][k]
    #
    #
    # np.save(f'figures/temp_data/words_probilities/{seq_len}/position_frequency_neg.npy', position_frequency_neg)
    #
    # return


    # position_frequency = np.load(f'figures/temp_data/words_probilities/{seq_len}/position_frequency_neg.npy')
    # norm_frequency = position_frequency / (len(valid_dataset)//2)
    # norm_frequency = minmax_scale(position_frequency, axis=1)
    # np.save(f'figures/temp_data/words_probilities/{seq_len}/position_frequency.npy', position_frequency)



# # # # # # # # # # # # # # # # plot words distribute # # # # # # # # # # # # # # #
#     position_frequency = np.load('position_frequency.npy')
#

# 画前十个词在每个位置的出现频率图
#     plt.figure(figsize=(60, 8))
#     for i in range(norm_frequency.shape[0]):
#         plt.subplot(2, 5, i + 1)
#         # plt.ylim(0.03, 0.06)
#         sns.barplot(x=np.arange(norm_frequency.shape[1]), y=norm_frequency[i], color='royalblue')
#         plt.title('weighted_neg')
#     plt.show()
#     plt.savefig(f'figures/weighted_neg.svg')
#
#     return

#
#     print(1)
# # # # # # # # # # # # # # # # plot words distribute # # # # # # # # # # # # # # #



    # [3444, 410, 3968, 2479, 220, 3524, 2157, 2812, 3435, 3502, 2911, 3750, 488, 3964, 1418, 55, 2373, 1824, 599, 2975, 1836, 3106, 2380, 1993, 4077, 1287, 871, 1511, 1563, 1615, 1472, 3805, 1011, 1593, 1869, 1648, 2641, 2326, 2085, 10, 1596, 3035, 354, 3839, 898, 508, 3505, 614, 1229, 511, 115, 2024, 2306, 578, 100, 1148, 1764, 247, 4012, 3393, 1885, 499, 921, 112, 1207, 2831, 1773, 172, 3419, 1912, 2964, 1503, 1570, 3461, 72, 2699, 1115, 553, 3530, 561, 470, 3428, 776, 3619, 847, 2903, 2550, 2011, 99, 4042, 3788, 1002, 3012, 3267, 2579, 2531, 4045, 3723, 655, 361, 3480, 4041, 1212, 1146, 1030, 2633, 3740, 74, 1825, 3677, 1871, 2571, 68, 226, 892, 123, 290, 1075, 827, 3728, 3467, 2988, 533, 1879, 2877, 1794, 3392, 805, 3967, 1829, 4003, 1103, 3227, 1587, 2644, 3636, 1055, 2149, 1753, 2518, 519, 2719, 1411, 2065, 2328, 1478, 1027, 3936, 3494, 2548, 1156, 3643, 2239, 2749, 767, 121, 1495, 338, 4050, 974, 2421, 324, 3946, 1467, 3356, 1363, 1926, 437, 2574, 3799, 1533, 2352, 2676, 1436, 2756, 2836, 3325, 381, 3942, 1487, 3119, 2088, 2336, 2109, 1059, 1282, 59, 3077, 2992, 2451, 2560, 1346, 2653, 581, 1585, 129, 382, 1893, 3503, 1722, 2718, 2861, 2754, 97, 2879, 1828, 1315, 2399, 2592, 387, 2164, 144, 467, 3219, 4031, 2488, 148, 3891, 2114, 3479, 2716, 3054, 4088, 3873, 3446, 1725, 2543, 2064, 1858, 3144, 3681, 2838, 3456, 5, 1208, 1965, 119, 1736, 2824, 2664, 1074, 3186, 2921, 1751, 2650, 2449, 3614, 3675, 2995, 3621, 3407, 2430, 479, 1677, 397, 1793, 2912, 2355, 1043, 1538, 3216, 1972, 3732, 2939, 3878, 2905, 3924, 2811, 169, 2627, 1369, 610, 2234, 200, 538, 1449, 507, 509, 2495, 4046, 3606, 2292, 2095, 3430, 2544, 2304, 3263, 3025, 787, 3765, 654, 2777, 3603, 30, 1847, 2460, 143, 3774, 1852, 1589, 3064, 2606, 3520, 904, 3940, 1181, 3963, 3757, 3122, 2225, 285, 3661, 956, 940, 1896, 3402, 1093, 2160, 1234, 2038, 2629, 707, 3340, 1996, 2934, 2758, 3500, 3780, 873, 3880, 3597, 3691, 3367, 1175, 3662, 1085, 2136, 3918, 2979, 1022, 445, 2224, 475, 2270, 2005, 3426, 730, 2807, 1494, 2710, 1063, 2054, 1799, 880, 4082, 3811, 3850, 2016, 1372, 1036, 2179, 1671, 550, 2724, 2982, 134, 3199, 2316, 268, 2526, 1624, 2063, 3211, 2705, 2916, 3638, 1894, 3557, 3983, 2198, 2193, 1523, 931, 3853, 3209, 2955, 969, 3425, 868, 3682, 3090, 2926, 209, 3554, 1543, 232, 2958, 2391, 109, 2680, 2429, 1138, 4004, 3277, 1508, 3278, 2889, 1165, 606, 1240, 2847, 1695, 1166, 2177, 1783, 1988, 3228, 520, 1576, 2226, 2966, 1536, 3273, 1556, 3743, 1801, 2059, 686, 3390, 2581, 3378, 2331, 3495, 3719, 3813, 891, 3171, 1789, 3943, 3933, 2952, 3630, 2647, 84, 3693, 512, 292, 2061, 2313, 1571, 652, 3240, 3083, 3204, 2279, 231, 1425, 3646, 1765, 2900, 2588, 1110, 2204, 2997, 887, 3752, 641, 895, 987, 1618, 1647, 3105, 1343, 3305, 1402, 498, 2612, 3652, 3066, 315, 1465, 3137, 3708, 2630, 1721, 2422, 3702, 2152, 3612, 2099, 2461, 3692, 2677, 3671, 1890, 1897, 225, 2620, 1935, 2432, 4066, 2227, 20, 1119, 2031, 1139, 398, 2440, 3362, 846, 544, 2259, 2709, 882, 76, 1931, 3501, 2938, 763, 1101, 3169, 1340, 3592, 275, 3146, 3835, 4060, 3473, 1554, 1830, 2181, 2917, 2790, 3089, 2994, 447, 3265, 346, 618, 3532, 3412, 2220, 167, 1468, 735, 3929, 3742, 3269, 3571, 105, 825, 2898, 2841, 3101, 839, 1320, 3117, 2280, 113, 685, 2368, 3395, 2428, 3817, 413, 2413, 591, 800, 1613, 3576, 186, 472, 670, 3459, 2621, 702, 1015, 2274, 1914, 1606, 1681, 999, 2595, 2185, 411, 1652, 3923, 401, 1269, 632, 185, 3690, 3917, 2772, 1349, 1856, 1233, 412, 3019, 3865, 2625, 1837, 2643, 821, 718, 3205, 753, 3586, 1364, 1114, 2019, 1501, 3196, 170, 1680, 1180, 2366, 2266, 3510, 4037, 1230, 271, 2116, 2662, 440, 3062, 80, 330, 3900, 3609, 3476, 1566, 1137, 773, 2200, 1580, 2128, 3253, 3329, 526, 3962, 3588, 2897, 2053, 1713, 87, 3274, 3127, 1630, 783, 2825, 3840, 638, 1396, 3406, 1227, 4089, 1458, 3744, 3593, 809, 2044, 946, 3526, 2155, 1960, 358, 1182, 468, 1841, 2242, 2600, 586, 2577, 1191, 2796, 1547, 2074, 521, 2791, 1692, 574, 2967, 1572, 2949, 1032, 3449, 1056, 667, 2500, 1548, 3231, 2367, 3628, 3443, 1855, 1679, 1477, 3992, 2219, 1392, 2786, 101, 1560, 3463, 692, 792, 3374, 1685, 823, 164, 3658, 2030, 1889, 1749, 3369, 1337, 3487, 1206, 1950, 874, 2382, 2076, 3626, 2329, 680, 1443, 2176, 3727, 2570, 2527, 2419, 3882, 2578, 1112, 1107, 2285, 2497, 3797, 1709, 2022, 2183, 1744, 3343, 2322, 2394, 2541, 1496, 419, 1232, 1389, 492, 3779, 3417, 1351, 1310, 1757, 2601, 3330, 1595, 1341, 3575, 2870, 2654, 40, 3521, 2409, 2582, 4011, 2501, 230, 2687, 1490, 91, 4005, 2598, 1822, 3072, 1716, 3577, 1541, 1684, 3257, 3158, 1745, 2141, 1854, 1953, 3152, 1951, 184, 3824, 1734, 3172, 2608, 2246, 865, 2318, 2517, 3468, 1336, 646, 2778, 2507, 3339, 966, 3081, 1605, 917, 370, 1281, 2121, 3351, 751, 35, 1553, 1743, 1908, 3408, 1842, 2091, 3810, 283, 3282, 1001, 3361, 3525, 2441, 2826, 234, 546, 2505, 1524, 1573, 71, 639, 3928, 693, 739, 2990, 369, 3802, 1561, 3308, 1130, 3176, 516, 3427, 2256, 549, 2264, 2442, 1738, 3489, 3844, 3042, 1661, 1816, 2587, 1295, 3082, 2360, 979, 717, 178, 1983, 2678, 2040, 3116, 389, 2639, 1886, 2738, 2783, 2703, 1562, 676, 2536, 997, 449, 3925, 879, 2340, 939, 3254, 86, 3620, 223, 233, 168, 106, 826, 1582, 366, 145, 2813, 2203, 3385, 1173, 1070, 1300, 2551, 4022, 1276, 2554, 1883, 3188, 1371, 2712, 262, 1293, 140, 1489, 662, 1123, 323, 3746, 250, 2691, 3046, 2787, 2818, 1057, 2130, 2702, 1532, 515, 3223, 944, 967, 3309, 142, 3496, 1772, 3399, 189, 3895, 1924, 2384, 2233, 2139, 1970, 1861, 1105, 3548, 428, 1733, 3080, 1742, 2941, 1029, 4038, 3633, 34, 2685, 360, 2731, 2100, 51, 612, 2821, 3213, 2249, 1108, 4091, 743, 265, 1409, 2093, 2801, 2673, 2689, 3591, 2833, 3969, 4007, 3071, 2433, 2947, 4014, 3635, 29, 1631, 1925, 267, 502, 3738, 3572, 2090, 348, 73, 1492, 2708, 2260, 3771, 1328, 833, 54, 434, 968, 3120, 1504, 4094, 659, 590, 239, 3099, 2514, 1863, 2484, 2797, 554, 64, 3178, 2895, 1705, 1682, 2502, 218, 2171, 3141, 227, 2312, 1626, 2087, 856, 1248, 3252, 43, 4048, 2503, 3433, 4047, 3759, 3132, 3237, 1149, 1262, 3947, 3018, 1767, 3551, 2640, 1766, 2940, 1519, 1756, 3509, 1221, 3715, 1399, 858, 899, 635, 2018, 2029, 543, 493, 2781, 769, 2082, 1422, 2341, 3140, 927, 3250, 934, 122, 1454, 1419, 926, 2262, 1517, 297, 1642, 3202, 2874, 1632, 3452, 1769, 2610, 0, 634, 2556, 4055, 778, 2142, 4072, 1226, 1375, 236, 327, 1474, 3051, 1575, 3672, 4049, 3600, 26, 4083, 3822, 971, 116, 2707, 420, 1201, 3793, 2800, 1699, 3903, 3021, 2416, 2023, 531, 2244, 4026, 3248, 3295, 1813, 1187, 2867, 154, 3934, 2692, 476, 727, 2021, 4057, 3030, 2470, 850, 39, 2999, 1607, 2844, 2729, 802, 1586, 3368, 3133, 17, 2153, 525, 1559, 2839, 3005, 1247, 1653, 3876, 2700, 3579, 6, 671, 3135, 945, 1804, 3826, 780, 1312, 2671, 564, 1482, 3347, 1237, 811, 2353, 197, 363, 2298, 1355, 3381, 3038, 3511, 1354, 1118, 202, 2642, 2344, 353, 163, 1748, 1930, 238, 415, 1622, 3888, 2129, 2480, 1521, 3230, 2187, 3229, 1374, 1393, 1800, 3153, 986, 1604, 1499, 206, 2696, 1689, 1985, 2400, 120, 156, 1218, 3808, 2003, 1275, 2043, 1067, 908, 1875, 335, 1567, 943, 804, 2275, 3647, 3722, 1949, 3126, 1701, 1592, 2679, 1746, 2102, 748, 3879, 537, 161, 3000, 3454, 146, 2077, 2058, 752, 388, 299, 1676, 203, 2207, 2690, 2904, 3403, 691, 1473, 1076, 1868, 452, 812, 600, 669, 3249, 256, 726, 291, 3569, 1614, 1486, 2196, 3420, 7, 46, 3562, 2615, 4073, 3346, 3373, 1203, 1471, 524, 1899, 2222, 1080, 13, 3868, 160, 3460, 2799, 2390, 2602, 3700, 2405, 1870, 1430, 3898, 1179, 367, 1028, 1456, 385, 928, 1761, 2257, 1210, 3292, 2123, 3782, 755, 1147, 3478, 404, 2737, 2496, 2538, 2884, 333, 2170, 1428, 1672, 201, 3611, 1012, 171, 2888, 2842, 1121, 657, 1786, 1034, 589, 4009, 3050, 4059, 1421, 1066, 217, 1771, 3896, 2299, 1895, 331, 1155, 1095, 2516, 2037, 1386, 3724, 3074, 2370, 801, 1014, 658, 3053, 2585, 719, 1183, 3806, 3659, 1905, 2308, 3637, 2439, 221, 1884, 2845, 3685, 75, 813, 3875, 2681, 2387, 604, 298, 3707, 1812, 2287, 3508, 2820, 1209, 4084, 1224, 704, 1126, 1920, 1690, 2932, 736, 3316, 2524, 2886, 624, 2775, 3131, 3190, 2348, 114, 3404, 2208, 994, 444, 222, 901, 1433, 1046, 837, 3264, 3550, 1117, 3218, 2363, 3266, 3326, 2931, 1796, 2564, 31, 2948, 421, 1919, 1188, 2138, 4030, 4018, 1329, 62, 3003, 985, 1939, 1546, 2733, 1542, 1265, 840, 1380, 608, 2546, 949, 3846, 1195, 469, 3656, 2017, 272, 3518, 3271, 356, 1457, 3332, 513, 2243, 471, 1301, 4040, 2154, 1876, 1569, 902, 3907, 2742, 224, 3270, 1408, 2978, 3095, 2078, 3052, 2120, 2798, 3418, 701, 3977, 2073, 4068, 2698, 1111, 750, 1096, 1054, 559, 616, 3970, 649, 249, 3695, 1516, 2115, 1509, 2605, 3344, 3008, 3242, 528, 3233, 3552, 3398, 1019, 2096, 1603, 2521, 1469, 2468, 747, 1005, 3447, 3115, 2552, 480, 2026, 276, 1125, 133, 563, 2626, 139, 1290, 3424, 3670, 1441, 1174, 1387, 2569, 2736, 1367, 864, 1403, 1683, 32, 295, 555, 305, 2549, 2182, 1072, 3288, 534, 2876, 181, 637, 3065, 2159, 2395, 3926, 287, 626, 781, 3320, 2773, 477, 1646, 2389, 396, 1338, 165, 3328, 1475, 816, 3564, 2184, 3108, 1698, 1809, 2167, 3349, 402, 1257, 955, 1862, 3396, 2358, 2869, 3995, 336, 3023, 1656, 819, 2025, 98, 992, 3382, 4020, 2922, 3483, 1545, 2346, 3544, 260, 1384, 2837, 2862, 4024, 3952, 38, 1039, 2052, 1729, 2276, 3807, 1707, 2856, 1973, 1213, 2403, 1133, 2860, 2596, 3166, 2349, 2573, 281, 2974, 4036, 2881, 494, 958, 1069, 3345, 41, 1696, 2757, 3022, 3457, 3867, 3154, 1164, 2072, 1120, 1193, 1045, 728, 2832, 2277, 575, 980, 2609, 96, 212, 1024, 3906, 2131, 3384, 3944, 1815, 2804, 3471, 594, 3416, 296, 355, 2547, 1099, 3415, 1997, 3541, 2188, 278, 2051, 3855, 2448, 1531, 2398, 3103, 3334, 270, 1158, 820, 756, 3775, 3111, 4002, 2638, 672, 2472, 2828, 2302, 61, 793, 1308, 2857, 1880, 2759, 2418, 894, 1416, 3232, 1470, 1539, 681, 3203, 3299, 2981, 1378, 57, 1714, 3859, 3047, 2178, 961, 2635, 1309, 2604, 1311, 2048, 859, 1330, 1933, 1106, 3570, 3076, 3198, 2873, 808, 52, 2586, 309, 1041, 1892, 2001, 1026, 2942, 3482, 644, 1008, 2950, 1955, 723, 2002, 2959, 286, 3971, 2944, 975, 1050, 1518, 3164, 746, 2850, 1485, 542, 3191, 3629, 1535, 2269, 2408, 2450, 2402, 423, 1819, 2887, 2776, 609, 1497, 3861, 1833, 383, 2985, 576, 196, 3324, 3280, 329, 2146, 1318, 2148, 1927, 518, 1432, 94, 3037, 568, 191, 1549, 2015, 1675, 3680, 500, 3244, 190, 4085, 1104, 1044, 1723, 2770, 3650, 3666, 3075, 1455, 1434, 3438, 1274, 3522, 598, 601, 3993, 2446, 439, 1216, 1827, 1654, 3490, 487, 174, 3654, 3848, 1688, 596, 82, 1990, 777, 194, 857, 2431, 339, 2401, 1333, 15, 3694, 2135, 2172, 2097, 788, 3988, 2205, 3173, 796, 3749, 2307, 2925, 890, 3776, 611, 1461, 1658, 3475, 1321, 3991, 386, 3147, 3513, 45, 1163, 1711, 4025, 3156, 2771, 8, 301, 674, 157, 3247, 482, 2923, 2963, 496, 3, 1989, 2882, 2147, 3812, 1420, 4063, 803, 3830, 3696, 2920, 2251, 2478, 1929, 2481, 602, 3583, 3919, 988, 3401, 724, 3357, 1178, 645, 441, 582, 1958, 3772, 708, 2784, 936, 1291, 2933, 3905, 2254, 491, 173, 2765, 3486, 932, 2345, 4095, 3110, 1073, 3376, 60, 2079, 124, 3632, 815, 2529, 2247, 50, 1186, 2134, 923, 3657, 3665, 1727, 1476, 3555, 1502, 3519, 1064, 304, 623, 3706, 683, 3709, 3726, 2859, 2751, 3318, 855, 2486, 1087, 1060, 2983, 489, 118, 2105, 2452, 1633, 2519, 2885, 1236, 2675, 3488, 2066, 3182, 2576, 458, 136, 3348, 3961, 2732, 1853, 2081, 3352, 2858, 3058, 1379, 372, 3668, 1442, 2083, 2050, 1693, 3506, 2238, 1062, 3911, 636, 3699, 1657, 3578, 2830, 1715, 2069, 948, 1040, 1966, 711, 334, 740, 963, 1197, 3410, 3581, 2271, 1484, 1923, 153, 3192, 1659, 1977, 1035, 1686, 2809, 2163, 779, 1466, 1674, 3179, 2511, 3916, 942, 884, 976, 474, 2376, 313, 110, 246, 3756, 1969, 1564, 2476, 734, 3215, 3289, 436, 2515, 254, 2070, 1358, 2730, 3932, 2875, 131, 1945, 443, 1220, 593, 442, 2301, 3987, 3291, 1790, 1192, 3261, 1544, 1750, 3437, 2245, 2106, 259, 2789, 1319, 208, 551, 2558, 2289, 266, 1959, 2268, 798, 1991, 2835, 1974, 1888, 807, 527, 2618, 2945, 3644, 2972, 866, 3948, 1792, 829, 2258, 628, 3177, 1460, 2880, 906, 2169, 1388, 1777, 1849, 960, 1018, 2780, 1305, 579, 1639, 3365, 952, 4052, 1000, 177, 3976, 1820, 2834, 282, 1272, 2906, 1323, 2417, 1397, 3210, 2388, 1304, 2530, 1992, 2462, 2385, 3258, 1798, 1288, 954, 3602, 688, 2555, 3323, 182, 1938, 1452, 3561, 1529, 2282, 294, 2810, 3634, 3109, 2212, 1097, 2878, 830, 3828, 1382, 1222, 1189, 3268, 3507, 1258, 3863, 1196, 2362, 3123, 3803, 762, 431, 1317, 2537, 3124, 1611, 2414, 1754, 3914, 3814, 1172, 2075, 162, 1902, 3061, 2667, 3729, 3687, 919, 1385, 558, 2661, 506, 2646, 965, 617, 677, 1136, 1802, 3870, 3200, 4000, 845, 957, 3432, 3272, 1339, 2004, 3093, 288, 1980, 757, 344, 252, 3599, 3786, 3363, 2545, 1763, 149, 132, 3565, 1857, 2263, 2357, 3165, 1978, 2378, 1946, 3796, 1762, 1599, 916, 1821, 1058, 2240, 881, 1649, 2216, 984, 213, 3441, 1190, 522, 310, 384, 193, 876, 1404, 848, 867, 3411, 251, 42, 2894, 159, 1609, 2132, 350, 1239, 3096, 3845, 1907, 2822, 2490, 1246, 3566, 2374, 1255, 321, 3466, 744, 198, 1512, 1417, 1527, 1818, 215, 4034, 2567, 455, 3667, 1303, 2036, 1127, 2648, 1141, 1017, 3073, 4035, 1007, 2607, 3056, 1669, 3007, 2071, 1645, 1313, 1814, 2961, 1129, 1446, 1719, 699, 990, 1741, 2107, 941, 2347, 1426, 462, 3547, 1911, 1302, 3899, 2000, 1826, 244, 3543, 572, 2829, 2232, 1344, 2721, 3978, 69, 2727, 3792, 1555, 3653, 2883, 1405, 326, 2977, 373, 2504, 4078, 1143, 2089, 720, 633, 70, 1253, 2617, 3831, 2350, 2645, 2943, 3372, 3920, 878, 1132, 1286, 3734, 37, 2561, 3499, 1169, 666, 3641, 2062, 2652, 3683, 973, 2794, 3909, 1779, 2872, 721, 584, 1481, 2704, 3296, 456, 883, 128, 14, 886, 2769, 3193, 1780, 2899, 3864, 2483, 332, 1922, 3938, 2935, 1483, 3514, 2457, 3251, 1967, 3113, 1177, 1568, 605, 3697, 2816, 571, 3580, 3832, 541, 463, 3760, 3704, 1913, 3197, 1979, 484, 1506, 585, 269, 3798, 682, 580, 2392, 3625, 2080, 3999, 2364, 1245, 1584, 2892, 2594, 725, 620, 2593, 1004, 3377, 2291, 2508, 1243, 375, 1352, 2915, 2686, 3027, 1534, 3174, 347, 1774, 3184, 3972, 406, 532, 1267, 1952, 2852, 1273, 308, 307, 374, 263, 977, 1795, 929, 2485, 836, 501, 3851, 2657, 3789, 909, 3605, 4092, 3300, 3651, 937, 127, 1932, 505, 2435, 451, 1423, 3989, 3314, 995, 1334, 4065, 4093, 535, 2186, 1401, 1270, 4039, 3856, 2217, 2161, 318, 715, 1083, 2092, 1061, 3618, 1198, 3886, 3587, 3002, 2914, 2273, 1917, 1594, 529, 1891, 3815, 24, 3028, 1650, 3721, 3493, 3866, 371, 3825, 1407, 352, 379, 771, 3375, 3689, 835, 2788, 1077, 2320, 2717, 1513, 1947, 1347, 3224, 3720, 3036, 58, 23, 95, 3276, 3874, 3387, 3857, 395, 425, 1578, 349, 3440, 1528, 1113, 2701, 3049, 2740, 2752, 3139, 3160, 915, 3472, 36, 345, 2649, 2145, 56, 3818, 465, 3364, 3168, 1089, 3462, 1184, 3128, 2634, 3516, 1128, 775, 2927, 1488, 2039, 3939, 2122, 832, 661, 359, 3714, 1211, 4079, 3341, 3725, 2290, 2865, 1013, 3823, 2805, 1010, 828, 1327, 1160, 2575, 3413, 3405, 1006, 3342, 3795, 1662, 3039, 1574, 3512, 3014, 1140, 3067, 3162, 2924, 2084, 3893, 3607, 3442, 911, 188, 1781, 2739, 3355, 2743, 2970, 1843, 3833, 1810, 3225, 1268, 3389, 2332, 2819, 703, 4058, 970, 28, 1431, 983, 629, 1168, 2248, 3608, 1395, 2311, 1052, 317, 1223, 2297, 438, 2957, 430, 3313, 2590, 2863, 570, 2764, 207, 3998, 394, 3622, 3434, 1805, 597, 258, 3431, 2162, 1655, 3221, 2372, 2325, 33, 2008, 3949, 280, 150, 2041, 556, 2849, 3032, 2361, 4067, 731, 2962, 2151, 3157, 2201, 3091, 3546, 1326, 2684, 2335, 351, 1982, 147, 1241, 673, 3975, 1278, 1636, 3201, 137, 1848, 2112, 1874, 872, 3383, 784, 2976, 48, 3819, 2334, 2111, 1342, 3414, 3031, 2447, 67, 2599, 1640, 4029, 1944, 982, 3889, 392, 3189, 3286, 2296, 1627, 996, 417, 11, 2656, 1092, 1152, 2909, 810, 1390, 1962, 896, 1406, 3645, 1135, 3748, 3930, 2067, 179, 3545, 2984, 1438, 77, 3301, 216, 1732, 782, 1620, 1840, 1909, 2659, 640, 1882, 426, 2553, 407, 3337, 2255, 1283, 2793, 460, 376, 3175, 3044, 1082, 1918, 3542, 125, 1263, 1463, 3678, 3017, 303, 3195, 1332, 2125, 1124, 2779, 3673, 391, 3559, 3070, 390, 400, 1788, 497, 2235, 19, 631, 1514, 2425, 4074, 3310, 2893, 3763, 1414, 853, 1726, 316, 2272, 3849, 1957, 3236, 3674, 3283, 1162, 687, 2583, 328, 90, 1047, 3877, 1597, 842, 1194, 2745, 1601, 1878, 2466, 1042, 3088, 4051, 1214, 1768, 2396, 849, 738, 2991, 126, 2614, 204, 3595, 2437, 2509, 1215, 1758, 2027, 1053, 1204, 274, 461, 2118, 1934, 3660, 2284, 577, 2572, 3059, 1279, 3730, 1400, 1410, 2864, 3912, 3617, 4080, 1199, 3996, 709, 1348, 1981, 433, 643, 924, 2342, 2969, 3985, 2013, 237, 2241, 3322, 3536, 3741, 2293, 2426, 3567, 1651, 2493, 2463, 2436, 2214, 483, 2323, 1635, 2068, 2168, 2049, 3163, 4081, 1219, 3698, 1752, 1242, 2343, 2956, 2288, 2445, 3167, 25, 2333, 3589, 3078, 663, 2189, 2668, 357, 705, 1775, 2597, 1817, 981, 3448, 3155, 566, 3843, 1462, 2354, 2854, 2371, 1373, 3183, 2591, 2150, 3034, 1877, 539, 277, 3829, 3335, 907, 905, 3966, 2209, 3470, 1391, 2722, 3041, 1956, 947, 2415, 3784, 1846, 1037, 3950, 66, 364, 548, 625, 155, 4, 834, 3753, 3703, 2110, 3515, 2475, 3872, 889, 1638, 2042, 1051, 3206, 3523, 2281, 1735, 789, 1928, 2295, 2144, 3327, 653, 2853, 3768, 3766, 4061, 3731, 279, 2910, 1867, 920, 1071, 2119, 3951, 253, 3852, 3639, 3754, 3129, 255, 3151, 745, 2228, 2694, 3553, 651, 1102, 3710, 1961, 427, 3086, 2902, 3686, 2817, 3114, 92, 2622, 1381, 3676, 3616, 903, 2339, 698, 3745, 1948, 2261, 1776, 3388, 1550, 2047, 378, 3048, 1307, 3649, 3574, 2012, 2908, 3984, 2929, 1167, 2688, 2199, 3315, 1906, 1335, 824, 2020, 3871, 713, 510, 151, 1231, 1628, 245, 1551, 1202, 1429, 2670, 933, 3148, 523, 1284, 3648, 3451, 1687, 1068, 2808, 2156, 158, 3370, 1297, 495, 3737, 695, 81, 1451, 2723, 229, 4056, 2175, 2843, 199, 716, 2453, 3107, 697, 514, 2195, 2407, 1667, 831, 3294, 3537, 3118, 2827, 530, 1964, 1637, 300, 3336, 2613, 1200, 3379, 3736, 3338, 311, 2192, 3767, 2338, 314, 603, 3921, 240, 2143, 88, 3185, 4010, 2173, 3094, 3885, 785, 3556, 689, 453, 2294, 1986, 4043, 2763, 3069, 2303, 3538, 2855, 1664, 3359, 3901, 607, 3011, 261, 1256, 2510, 2937, 2351, 3973, 2379, 405, 424, 877, 4053, 1145, 3246, 117, 503, 2491, 16, 1023, 1205, 799, 3063, 243, 1558, 1522, 2953, 2624, 2672, 2753, 1778, 2474, 3534, 2665, 1916, 322, 1259, 1316, 3307, 2045, 2623, 1260, 3785, 3604, 3527, 613, 630, 3238, 1003, 478, 3161, 4032, 2848, 1157, 3960, 2237, 3043, 1154, 2589, 4023, 2580, 749, 3642, 2755, 1185, 1811, 1098, 1851, 3033, 1134, 1090, 1109, 737, 3862, 2871, 1976, 1394, 380, 1540, 1665, 1823, 690, 2989, 4027, 3539, 2035, 1670, 4008, 1600, 900, 1901, 893, 1984, 1697, 2365, 768, 1881, 3371, 1507, 3974, 2506, 1250, 3409, 235, 1623, 852, 3800, 569, 3894, 3354, 187, 1350, 1872, 2213, 4033, 3624, 2124, 1915, 2744, 2714, 710, 722, 1702, 1740, 1838, 2814, 2410, 3239, 3540, 1398, 2666, 4021, 3290, 595, 2919, 2046, 293, 3558, 242, 1943, 210, 2711, 2683, 1785, 794, 3529, 562, 844, 3214, 2528, 3465, 108, 854, 403, 3573, 1280, 3259, 3222, 2086, 3982, 1383, 615, 817, 3497, 2265, 3150, 770, 2540, 3517, 3747, 3145, 1142, 1493, 2960, 3319, 592, 2250, 1360, 195, 2310, 3284, 2494, 1954, 1807, 1643, 2165, 2695, 2557, 3770, 1447, 3568, 1084, 1598, 978, 4017, 3945, 3598, 3016, 3935, 1362, 3085, 3304, 1368, 3904, 2523, 1031, 3549, 2492, 3220, 838, 3601, 2191, 1673, 2215, 2766, 450, 913, 3809, 2651, 3958, 1616, 228, 950, 2094, 4013, 795, 3125, 3180, 1755, 3235, 2951, 464, 733, 3207, 565, 306, 2427, 4075, 1415, 1366, 2660, 2513, 3860, 3020, 219, 2032, 1621, 1694, 2525, 3712, 319, 3394, 1439, 2565, 1608, 3386, 176, 991, 540, 65, 1835, 3455, 2327, 679, 1254, 1244, 3827, 3358, 2785, 3758, 2891, 860, 2469, 3931, 1038, 1009, 2321, 2987, 2033, 2616, 706, 1557, 3711, 3777, 1968, 2726, 2768, 248, 3136, 754, 3333, 416, 2986, 2693, 2236, 3453, 2725, 3170, 1904, 47, 3057, 1491, 2223, 473, 3321, 1581, 2795, 545, 3241, 2728, 3640, 3869, 1873, 2113, 4016, 587, 3134, 1150, 1, 2697, 2774, 3255, 660, 2489, 3130, 989, 1424, 3842, 2381, 3391, 2, 21, 3060, 2482, 3804, 3275, 2356, 2465, 3887, 3331, 1370, 2434, 3590, 851, 2286, 1081, 2034, 1591, 3764, 457, 340, 1296, 3762, 2802, 183, 3791, 3908, 1228, 3079, 3481, 214, 1079, 1612, 557, 1975, 4006, 3087, 1704, 2720, 2314, 3623, 1357, 1292, 4090, 664, 53, 1159, 3450, 1759, 1590, 3243, 408, 1306, 1987, 3937, 1937, 2913, 560, 1176, 1663, 3208, 3485, 1834, 3716, 93, 2806, 3718, 2397, 2300, 2369, 3669, 2377, 1710, 1324, 2126, 205, 1453, 2563, 2383, 1730, 3226, 1091, 399, 1998, 2735, 2007, 700, 1832, 885, 2056, 3535, 648, 1737, 3281, 2535, 861, 3834, 772, 3913, 3615, 3790, 1995, 4019, 935, 3627, 3423, 862, 1153, 3015, 1116, 1078, 3436, 3533, 2532, 377, 2211, 3045, 3010, 3138, 2631, 3531, 642, 2253, 1445, 2375, 1353, 1217, 9, 1588, 3302, 760, 1887, 78, 1700, 2455, 1770, 843, 1325, 1459, 1666, 621, 2406, 3783, 241, 2283, 870, 2166, 3303, 3293, 665, 3994, 3429, 1839, 1942, 486, 2231, 1376, 1644, 2534, 2767, 2851, 2330, 1480, 2973, 3892, 3159, 3883, 3890, 264, 2218, 362, 3769, 446, 2993, 2098, 1530, 1020, 3563, 1413, 1122, 2471, 1526, 1450, 1864, 2628, 337, 1298, 3858, 3484, 3655, 3684, 3821, 3092, 1900, 454, 1049, 2815, 2520, 102, 2319, 2713, 3024, 3585, 1515, 2267, 2674, 3498, 3234, 257, 656, 925, 4015, 3915, 650, 3582, 3422, 180, 2559, 138, 3097, 3279, 1994, 2840, 409, 818, 1936, 678, 2930, 1048, 1791, 1831, 44, 3477, 3312, 998, 418, 3439, 4028, 3121, 1322, 3006, 2658, 2459, 914, 1720, 888, 2746, 3469, 3801, 3142, 1131, 2458, 2715, 2998, 938, 3040, 2140, 85, 3187, 1144, 343, 2901, 3287, 732, 2762, 668, 341, 135, 1610, 2127, 567, 3965, 3717, 2028, 1619, 3794, 1712, 104, 3560, 1706, 875, 972, 3013, 3631, 3986, 1289, 2424, 3910, 3980, 1717, 1728, 2499, 2473, 4070, 3751, 1065, 83, 289, 1520, 325, 3664, 2663, 2562, 3260, 1903, 1941, 89, 3491, 3366, 2968, 1271, 3100, 3285, 3143, 2467, 1412, 759, 1299, 758, 2456, 211, 1444, 3353, 3397, 1921, 1331, 3837, 2055, 273, 2133, 765, 3421, 2542, 3317, 1708, 1235, 2954, 1787, 552, 2936, 2760, 1803, 4086, 2706, 175, 3957, 1238, 573, 2611, 284, 869, 536, 1703, 1963, 1170, 3854, 2965, 3979, 1498, 1866, 684, 696, 1602, 2632, 1277, 2464, 2946, 761, 448, 1025, 3360, 3881, 2158, 365, 2896, 2890, 4071, 814, 588, 2477, 1171, 2180, 1731, 2324, 951, 1678, 953, 2539, 302, 3380, 1094, 3679, 1797, 791, 714, 912, 2682, 766, 2252, 1266, 2009, 1691, 3739, 79, 1660, 1359, 1999, 1314, 1249, 466, 2971, 2305, 2533, 2315, 2101, 1088, 3941, 490, 3701, 49, 962, 3997, 2060, 3306, 18, 694, 1356, 729, 2411, 4076, 2823, 3778, 2996, 422, 2980, 2006, 2734, 1285, 3688, 1161, 3663, 1525, 517, 3098, 312, 3902, 3847, 841, 3773, 414, 1806, 3781, 2761, 2206, 2868, 3594, 2568, 3981, 2174, 1724, 3068, 922, 2423, 1377, 1808, 3733, 432, 2359, 1971, 459, 1629, 2014, 3956, 3298, 2866, 3102, 393, 2438, 3400, 2636, 3084, 742, 964, 1565, 1859, 1760, 1086, 2637, 1898, 583, 342, 1435, 1505, 2108, 2337, 647, 774, 2057, 863, 1537, 959, 2197, 1747, 1641, 3445, 822, 141, 2221, 2420, 3297, 1583, 2907, 1016, 1100, 485, 2444, 3474, 1033, 2103, 3262, 1860, 1437, 22, 3026, 3112, 1464, 2747, 3104, 3029, 3596, 2619, 2278, 3245, 3922, 806, 3761, 3256, 2229, 1739, 3959, 4001, 3001, 1579, 1251, 3004, 2750, 1784, 2669, 2309, 1225, 2137, 675, 3458, 4044, 2748, 3613, 786, 3705, 3927, 790, 1479, 1365, 2386, 3492, 3954, 2928, 622, 3350, 3528, 368, 2104, 3735, 2522, 1668, 1510, 4064, 1625, 1617, 1252, 4054, 4062, 910, 2566, 2190, 2512, 2317, 1261, 930, 3836, 3311, 1634, 103, 918, 2230, 712, 2443, 3990, 1865, 2194, 2454, 797, 2792, 3838, 27, 2918, 1845, 1718, 2210, 3584, 1850, 130, 2487, 2803, 152, 1361, 320, 1151, 435, 1844, 547, 3897, 3755, 3212, 3464, 1910, 2584, 3820, 166, 481, 192, 2846, 897, 1940, 3194, 3841, 4069, 2117, 3787, 12, 2393, 764, 1264, 619, 3149, 2498, 1577, 1500, 4087, 1294, 3217, 3610, 3504, 2202, 1552, 429, 3816, 2404, 2603, 3009, 111, 741, 3953, 2741, 3181, 627, 1782, 107, 3713, 504, 63, 2010, 3955, 1448, 2782, 1345]
    # words = np.concatenate(words)
    # words_flatten = words.reshape(-1)
    # counts = Counter(words_flatten)
    # # total_counts = sum(counts.values())
    # top_240 = counts.most_common(240)
    #
    #
    # x_data = np.arange(240)
    # y_data = [top_id[-1] for top_id in top_240]
    # plt.figure(figsize=(18, 6))
    # plt.title("level bar")
    # plt.ylabel("counts")
    # plt.xlabel("level")
    # plt.bar(x_data + 0.8, y_data, width=0.5)
    # plt.axis([0, 240, 0, 6000])
    # plt.show()


    ############################################## position frequence analyse ##########################################
    # frequecy_word_rank = [3583, 653, 4078, 2743, 2530, 3232, 3296, 1353, 3432, 3711, 848, 7, 2660, 3541, 453, 2222, 2030, 1222, 3776, 690, 3878, 2969, 1082, 2102, 2390, 654, 2601, 3859, 2766, 2483, 2778, 3005, 1323, 2962, 238, 1179, 3718, 1094, 3334, 1864, 2705, 1540, 2572, 1960, 2380, 3621, 2780, 331, 3998, 1940, 474, 1669, 963, 1993, 2892, 312, 2657, 3591, 3914, 1849, 1106, 2004, 1391, 196, 1519, 2820, 2624, 3648, 860, 879, 3405, 723, 52, 974, 2070, 3088, 2246, 1271, 2972, 2888, 811, 2613, 710, 1090, 2549, 1964, 3020, 3515, 219, 725, 157, 2003, 2898, 4058, 59, 3844, 1560, 361, 1930, 3135, 4043, 609, 541, 869, 3079, 181, 2946, 1493, 276, 825, 1910, 2518, 698, 1030, 263, 1659, 3285, 2866, 2956, 3725, 2052, 1348, 1175, 3071, 3227, 678, 324, 2283, 2310, 2227, 4041, 899, 3659, 2211, 2345, 756, 3031, 1213, 2256, 1395, 2636, 3756, 1112, 119, 805, 1734, 2476, 775, 2270, 2753, 3702, 1997, 1012, 1203, 800, 2251, 969, 55, 56, 1933, 746, 208, 478, 2672, 737, 674, 1456, 163, 443, 3805, 2529, 3923, 1380, 3693, 3457, 2950, 1182, 749, 2762, 3661, 3559, 728, 2316, 3362, 183, 933, 4055, 1797, 2431, 2573, 2042, 1091, 1265, 1315, 2480, 2110, 16, 2172, 1947, 917, 3084, 3973, 2894, 2662, 342, 1126, 1471, 865, 4012, 399, 1851, 297, 893, 1248, 1794, 3913, 375, 1457, 1731, 776, 3517, 2327, 3775, 1307, 669, 1196, 2673, 3037, 3684, 3198, 3733, 539, 1547, 2958, 2277, 788, 2257, 931, 466, 861]

# 画前十个词的motif
def draw_motif(model, top_10, name):
    fig, axes = plt.subplots(2, 5, figsize=(10,3),dpi=300)
    plt.subplots_adjust(wspace=0.3)
    pwm = model.detect_word[0].weight[:,:,:].cpu().detach().numpy()

    pwm = pwm - np.mean(pwm, axis=1, keepdims=True)
    pwm = pwm - np.abs(pwm).mean(1, keepdims=True)*0.7
    # for ii, m in enumerate(frequecy_word_rank[:5]):
    #     plotfun(pwm[m][::-1, ::-1].T, ax=axes[ii], title=m,
    #             ylabel=True if ii == 0 else False)

    for ii, m in enumerate(top_10):
        if ii < 5:
            plotfun(pwm[m].T, ax=axes[0][ii], title=m,
                    ylabel=True if ii == 0 else False)
        else:
            plotfun(pwm[m].T, ax=axes[1][ii-5], title=m,
                    ylabel=True if ii == 0 else False)
    plt.title(name)
    plt.savefig(f"./figures/{name}.svg")
    plt.show()


    ################################################### attention map #####################################################

def main():
    params = {
        'kernel_num': 4096,
        'topk': 128,
        'lr': 0.0001,
        'batch_size': 128,
        'epoch': 100,
        'seq_len': 501,
        'saved_model_name': 'diff_len_',
        'seed': 17,
    }
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