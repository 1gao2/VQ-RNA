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
import glob
from sklearn.preprocessing import MinMaxScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


def draw_heatmap(score, name, attention_type):
    plt.figure(figsize=(12, 10))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    # plot
    sns.set()
    # scores = (scores - scores.min()) / ( scores.max() - scores.min())
    # ax = sns.heatmap(scores, cmap='Greens')
    ax1 = sns.heatmap(score, cmap='YlGnBu')
    if attention_type == 'all':
        plt.savefig(f'attention/all_layer&head_attentions/{name}.svg')
    elif attention_type == 'last':
        plt.savefig(f'attention/last_layer&all_head_attentions/{name}.svg')

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
        output, _= net(data)

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
    if attention_type == 'last':
        attention = representation[-1]
        attn = format_attention(attention)
        attn_score = torch.sum(attn, dim=1).squeeze()  # last layer & all head

    elif attention_type == 'all':
        attention = representation
        attn = format_attention(attention)
        attn_score = torch.sum(torch.sum(attn, dim=0).squeeze(), dim=0).squeeze()  # all layer & all head

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

def my_evaluation_method(params):

#################################################### prepare data #####################################################
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
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

################################################### prepare data ######################################################

####################################################### train #########################################################
    # saved_model_name = f'm7g_attetion_{seq_len}'
    #
    # batch_size_r = 256
    # lr_r = 0.0001
    # epoch_r = 100
    # #
    # best_acc = rna_encoder(train_loader, valid_loader, test_loader,  lr_r, epoch_r, batch_size_r, saved_model_name)
####################################################### train #########################################################

################################################### attention map #####################################################

    model = Lucky(kernel_num=params['kernel_num'], topk=params['topk'])
    # best_acc = 0.8961272891023716
    # load_params(model, config.path_params)
    model.load_state_dict(torch.load(f'save/mymodel/data1/seed2.pth'))
    # model.load_state_dict(torch.load(f'saved_models/401_0.840929808568824.pth'))
    # model.load_state_dict(torch.load(f'saved_models/301_0.8350045578851413.pth'))
    # model.load_state_dict(torch.load(f'saved_models/201_0.8158614402917046.pth'))
    # model.load_state_dict(torch.load(f'saved_models/101_0.8113035551504102.pth'))
    # model.load_state_dict(torch.load(f'saved_models/51_0.7948951686417502.pth'))
    model.eval()
    model = model.to(device)

    test_performance, test_roc_data, test_prc_data = evaluate(valid_loader, model)
    print(test_performance)

    scores_1024_avg = []
    # attention_type = 'all'
    attention_type = 'last'

    test_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    for seq, label in tqdm(test_loader):
        seq, label = seq.to(device), label.to(device)
        # train_attention = train_attention.to(device)
        logits, attention_1024 = model(seq)
        # print(attention_1024[0].shape)
        # scores_1024 = visualize_one_sequence(attention_1024)
        scores_1024 = np.array(full_attention(attention_1024, attention_type).cpu().detach())
        # scores_510_avg += scores_510
        scores_1024_avg.append(scores_1024)

    # print(scores_510_avg)
    # print(scores_1024_avg)


    scores_1024_avg = np.sum(np.stack(scores_1024_avg, axis=0), axis=0)
    # scores_510_avg = minmax_scale(scores_510_avg, axis=1)
    scores_1024_avg = minmax_scale(scores_1024_avg, axis=1)
    # scores_1024_avg = minmax_scale(scores_1024_avg.flatten())
    # print(scores_510_avg)
    # print(scores_1024_avg)

    # draw_heatmap(scores_510_avg, type='510')
    draw_heatmap(scores_1024_avg, seq_len, attention_type)


################################################### attention map #####################################################

def main():
    params = {
        'kernel_num': 4096,
        'topk': 128,
        'lr': 0.0001,
        'batch_size': 128,
        'epoch': 100,
        'seq_len': 51,
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