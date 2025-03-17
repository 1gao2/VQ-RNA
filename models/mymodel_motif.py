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


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio=2, fused=True):
        super(ConvBlock, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv1d(inp, hidden_dim, 9, 1, padding=4, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=False),
            nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm1d(oup),
        )

    def forward(self, x):
        return x + self.conv(x)


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len).cuda()
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float().cuda()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, ):
        # self.encoding
        # [max_len = 512, d_model = 512]

        # [batch_size = 128, seq_len = 30]

        return self.encoding[:167, :].cuda()
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)

    def forward(self, inputs):
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape

        flat_input = inputs.view(-1, self._embedding_dim)

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        #print((quantized - inputs).mean().item())

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized = quantized.permute(0, 2, 1).contiguous()

        return vq_loss, quantized, perplexity, encodings

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

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
        x = F.relu(self.conv1(inputs.float()))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose1d(in_channels=64, out_channels=512, kernel_size=4, stride=2, padding=1)

        # 第二层反卷积：进一步恢复序列长度
        self.deconv2 = nn.ConvTranspose1d(in_channels=512, out_channels=512 // 2, kernel_size=4, stride=2, padding=1)

        # 第三层反卷积：输出重构的 DNA 序列
        self.deconv3 = nn.ConvTranspose1d(in_channels=512 // 2, out_channels=4, kernel_size=4, stride=3, padding=2)


    def forward(self, inputs):
        x = F.relu(self.deconv1(inputs))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)

        return x

class Lucky(nn.Module):
    def __init__(self):
        """
        Parameters
        ----------
        """
        super(Lucky, self).__init__()


        self.positional_encoding = PositionalEncoding(256, 167)

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


        self._residual_stack2 = ResidualStack(in_channels=256,
                                             num_hiddens=256,
                                             num_residual_layers=2,
                                             num_residual_hiddens=32)

        self._encoder = Encoder()
        self._vq_vae = VectorQuantizer(num_embeddings=512, embedding_dim=64, commitment_cost=0.25)
        self._decoder = Decoder()


    def forward(self, x):
        """Forward propagation of a batch.
        """
        # x = torch.IntTensor
        x = x.to(torch.int64)
        # x = F.one_hot(x, num_classes=4).transpose(1, 2).float()

        z = self._encoder(x)
        x_cnn = z

        vq_loss, quantized, perplexity, _ = self._vq_vae(z)
        # quantized = self._residual_stack2(quantized)
        x_recon = self._decoder(quantized)
        x_recon = x_recon.transpose(1, 2).float()
        x_recon = torch.softmax(x_recon, dim=-1)
        x_recon = x_recon.squeeze(1)
        x_cnn = x_cnn.transpose(1, 2).float()
        quantized = quantized.transpose(1, 2).float()
        x = quantized


        atts = []
        for layer in self.layers:
            x, att = layer(x, None)
            atts.append(att)

        x_final = x[:, 21, :]
        out = self.final(x_final)

        return out, atts, vq_loss, x_recon, perplexity
        #return out, atts, vq_loss, x_recon, perplexity, x_final


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