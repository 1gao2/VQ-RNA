#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from models.transformer import EncoderLayer


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)

    def forward(self, inputs):
        # Change input dimensions for processing
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
        # First convolutional layer: Expands DNA sequence input from 4 channels to more feature channels
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=512 // 2, kernel_size=4, stride=2, padding=1)

        # Second convolutional layer: Further increases feature channels and reduces sequence length
        self.conv2 = nn.Conv1d(in_channels=512 // 2, out_channels=512, kernel_size=4, stride=2, padding=1)

        # Third convolutional layer: Converts output into a compact latent space representation
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=64, kernel_size=4, stride=3, padding=2)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # First transposed convolution: Expands feature dimension and starts restoring sequence length
        self.deconv1 = nn.ConvTranspose1d(in_channels=64, out_channels=512, kernel_size=4, stride=2, padding=1)

        # Second transposed convolution: Further restores sequence length
        self.deconv2 = nn.ConvTranspose1d(in_channels=512, out_channels=512 // 2, kernel_size=4, stride=2, padding=1)

        # Third transposed convolution: Outputs reconstructed DNA sequence
        self.deconv3 = nn.ConvTranspose1d(in_channels=512 // 2, out_channels=4, kernel_size=4, stride=3, padding=2)

    def forward(self, inputs):
        x = F.relu(self.deconv1(inputs))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)

        return x


class VQRNA(nn.Module):
    def __init__(self):
        """
        Parameters
        ----------
        """
        super(VQRNA, self).__init__()

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
        x = F.one_hot(x, num_classes=4).transpose(1, 2).float()
        z = self._encoder(x)
        vq_loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        x_recon = x_recon.transpose(1, 2).float()
        x_recon = torch.softmax(x_recon, dim=-1)
        x_recon = x_recon.squeeze(1)
        quantized = quantized.transpose(1, 2).float()
        x = quantized

        atts = []
        for layer in self.layers:
            x, att = layer(x, None)
            atts.append(att)
        x_final = x[:, 9, :]
        out = self.final(x_final)

        return out, atts, vq_loss, x_recon, perplexity


if __name__ == '__main__':
    VQRNA = VQRNA()
    VQRNA = VQRNA.cuda()
    VQRNA.eval()
    VQRNA = VQRNA.double()

    x = torch.randint(0, 4, (1, 501)).cuda()

    print(x)
    y = VQRNA(x)
    print(y)
    print(y.shape)
    print(y.dtype)
    print(y.device)
