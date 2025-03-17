#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
# from scipy.special import softmax
from matplotlib import pyplot as plt
# from selene_dataloader import SamplerDataLoader
from torch import nn
from models.transformer import EncoderLayer
import os


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 第一层卷积：将 DNA 序列的输入从 4 个通道扩展到较多的特征通道
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=512 // 2, kernel_size=4, stride=2,
                               padding=1)

        # 第二层卷积：进一步增加特征通道数，减小序列长度
        self.conv2 = nn.Conv1d(in_channels=512 // 2, out_channels=512, kernel_size=4, stride=2,
                               padding=1)

        # 第三层卷积：将输出转化为较紧凑的潜在空间表示
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=64, kernel_size=4, stride=3, padding=2)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x


class Lucky(nn.Module):
    def __init__(self):
        """
        Parameters
        ----------
        """
        super(Lucky, self).__init__()

        self._encoder = Encoder()

        self.layers = nn.ModuleList([EncoderLayer(d_model=64,
                                                  ffn_hidden=64,
                                                  n_head=8,
                                                  drop_prob=0.2)
                                     for _ in range(6)])

        self.final = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 2),
        )


    def forward(self, x):
        """Forward propagation of a batch.
        """
        x = F.one_hot(x, num_classes=4).transpose(1, 2).float()
        #x = F.one_hot(x, num_classes=4)

        x = self._encoder(x)
        x = x.transpose(1, 2).float()

        atts = []
        for layer in self.layers:
            x, att = layer(x, None)
            atts.append(att)
        # print(word_embedding.shape)

        # x_final = x.reshape(x.size(0), -1)
        # out = self.final_large(x_final)
        x_final = x[:, 21, :]
        out = self.final(x_final)

        #return out, atts
        return out, atts, x_final

if __name__ == '__main__':
    Lucky = Lucky()
    Lucky = Lucky.cuda()
    Lucky.eval()
    Lucky = Lucky.double()
    # x = torch.randn(1, 100000, 4).cuda()
    # x = torch.zeros(1, 4, 100000).cuda()
    # x = torch.randn(1, 4, 100000).cuda()
    x = torch.randint(0, 4, (1, 501)).cuda()
    # covert into one-hot encoding
    # x = F.one_hot(x, num_classes=4).transpose(1, 2).double()
    print(x)
    # x = x.double()
    # x = x.double()
    y = Lucky(x)
    print(y)
    print(y.shape)
    print(y.dtype)
    print(y.device)