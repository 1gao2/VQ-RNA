#!/usr/bin/env Python
# coding=utf-8
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
from models.mymodel import Lucky

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
    datas_neg = pd.read_csv(f"data/other/{data_type}/{file_index}-0.csv")
    datas_pos = pd.read_csv(f"data/other/{data_type}/{file_index}-1.csv")
    seq = list(datas_neg['data']) + list(datas_pos['data'])
    label = list(datas_neg['label']) + list(datas_pos['label'])

    seq = [s.replace(' ', '').replace('U', 'T') for s in seq]

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

def to_log(log, params):
    with open(f"results/mymodel/data{params['data_index']}/seed{params['seed']}111.log", "a+") as f:
        f.write(log + '\n')

# ========================================================================================

def train_model(train_loader, valid_loader, test_loader, params ,data_variance):
    # Define model
    model = Lucky().to(device)

    # Optimizer and loss
    opt = optim.Adam(model.parameters(), lr=params['lr'])
    criterion_CE = nn.CrossEntropyLoss()
    criterion_BCE = nn.BCELoss()
    best_acc = 0
    patience = params['patience']
    now_epoch = 0
    loss1 = params['loss1']
    loss2 = params['loss2']
    loss3 = params['loss3']
    best_model = None
    for epoch in range(params['epoch']):
        model.train()
        loss_ls = []
        t0 = time.time()
        for seq, label in train_loader:
            seq, label = seq.to(device), label.to(device)
            logits, _, vq_loss, data_recon, perplexity = model(seq)
            loss_CE = criterion_CE(logits, label)

            #recon_error = F.cross_entropy(data_recon.float(), seq.float()) / data_variance
            # recon_error = F.binary_cross_entropy_with_logits(data_recon.float(), seq.float()) / data_variance
            #data_recon = F.one_hot(data_recon, num_classes=4)
            seq = F.one_hot(seq, num_classes=4)
            #data_recon = F.one_hot(data_recon, num_classes=4)
            recon_error = F.mse_loss(data_recon.float(), seq.float())
            #recon_error = criterion_CE(data_recon.float(), seq.float())
            loss = loss_CE*loss1 + vq_loss*loss2 + recon_error*loss3
            # loss = loss_CE
            #print(f"loss_CE: {loss_CE.item()}, recon_error: {recon_error.item()}, vq_loss: {vq_loss.item()}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_ls.append(loss.item())

        # Validation step (if needed)
        model.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_prc_data = evaluate(train_loader, model)
            valid_performance, valid_roc_data, valid_prc_data = evaluate(valid_loader, model)

        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, vq_loss: {vq_loss:.5f}, recon_error: {recon_error:.5f} ,loss_CE: {loss_CE:.5f}\n"
        results += f'Train: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 16 + ' Valid Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                   + '\n[ACC, \tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            valid_performance[0], valid_performance[1], valid_performance[2], valid_performance[3],
            valid_performance[4], valid_performance[5]) + '\n' + '=' * 60
        valid_acc = valid_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
        print(results)
        to_log(results, params)

        if valid_acc > best_acc:
            best_acc = valid_acc
            now_epoch = 0
            best_model = copy.deepcopy(model)
            to_log('here occur best!\n', params)

            checkpoint = {
                'state_dict': model.state_dict()}

            torch.save(checkpoint, f"save/mymodel/data{params['data_index']}/seed{params['seed']}.pth")

        else:
            now_epoch += 1
            print('now early stop target = ', now_epoch)
        test_performance, test_roc_data, test_prc_data = evaluate(test_loader, model)
        test_results = '\n' + '=' * 16 + colored(' Test Performance. Epoch[{}] ', 'red').format(
            epoch + 1) + '=' * 16 \
                       + '\n[ACC,\tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            test_performance[0], test_performance[1], test_performance[2], test_performance[3],
            test_performance[4], test_performance[5]) + '\n' + '=' * 60
        print(test_results)
        to_log(test_results, params)

        if now_epoch > patience:
            print('early stop!!!')
            best_performance, best_roc_data, best_prc_data = evaluate(test_loader, best_model)
            best_results = '\n' + '=' * 16 + colored(' Test Performance. Early Stop ', 'red').format(
                epoch + 1) + '=' * 16 \
                           + '\n[ACC,\tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                best_performance[0], best_performance[1], best_performance[2], best_performance[3],
                best_performance[4], best_performance[5]) + '\n' + '=' * 60
            print(best_results)
            to_log(best_results, params)
            break

        #return best_acc


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


def evaluation_method(params):

    # for i in range(1, 11):
    train_x, train_y = read_file(data_type='train', file_index=params['data_index'])
    valid_x, valid_y = read_file(data_type='valid',file_index=params['data_index'])
    test_x, test_y = read_file(data_type='test',file_index=params['data_index'])


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

    data_variance = np.var(train_x)

    train_model(train_loader, valid_loader, test_loader, params, data_variance)


def main():

    for index in range(1, 2):
        params = {
            'lr': 0.001,
            'batch_size': 32,
            'epoch': 300,
            'seq_len': 501,
            'saved_model_name': 'diff_len_',
            'seed': 2,
            'data_index': index,
            'patience': 30,
            'loss1': 0.05819746775986198,
            'loss2': 0.04187829891956044,
            'loss3': 0.6804473004341116
        }

        seed = params['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        evaluation_method(params)

if __name__ == '__main__':
    main()