import pandas as pd
from tqdm import tqdm
import itertools
import numpy as np
import torch
tqdm.pandas(ascii=True)
import os
import sys
sys.path.append('../')
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tangermeme.ism import saturation_mutagenesis
import matplotlib.pyplot as plt
from plot import plot_logo
from VQRNA_motif import VQRNA
0

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    'batch_size': 32,
    'data_index': 1,
    'seq_len': 501
}

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
                seq.append(line.replace('T', 'U'))

    return seq, label

def read_file(data_type, file_index):
    datas = f"../data/motif_test/motif_test.fasta"
    seq, label = read_fasta(datas)

    return seq, label

def encode_sequence_1mer(sequences, max_seq):
    k = 1
    overlap = False

    all_kmer = [''.join(p) for p in itertools.product(['A', 'C', 'G', 'U', '-'], repeat=k)]
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

def load_model_and_data():
    """
    Load the model and data.
    """
    # Load data
    test_x, test_y = read_file(data_type='test', file_index=params['data_index'])
    test_x, test_y = np.array(test_x), np.array(test_y)
    test_x = encode_sequence_1mer(test_x, max_seq=params['seq_len'])

    test_dataset = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, drop_last=True)

    # Load the model
    model = VQRNA().to(device)
    model.load_state_dict(torch.load(f"../saved_models/draw_motif_seed2.pth")['state_dict'])

    return model, test_loader, test_dataset


class SliceWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SliceWrapper, self).__init__()
        self.model = model
        # self.target = target

    def forward(self, X):
        out = self.model(X)
        out = out[0][:, 1].unsqueeze(1)
        # print(out.shape)
        result = torch.sigmoid(out)
        #print(result)
        return result


model, test_loader, test_dataset = load_model_and_data()
wrapper = SliceWrapper(model).cuda()

# "Initialize the accumulated feature importance matrix"
total_attr1 = None
total_attr2 = None
total_attr3 = None
total_attr4 = None
total_attr5 = None
total_attr6 = None

for i in range(len(test_dataset)):

    x = test_dataset[i][0].unsqueeze(dim=0)
    x = F.one_hot(x, num_classes=4).transpose(1, 2).float()

    # "Calculate feature importance"
    X_attr1 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=250 - 50, end=250 + 50, device='cpu')
    X_attr2 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=50 - 50, end=50 + 50, device='cpu')
    X_attr3 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=150 - 50, end=150 + 50, device='cpu')
    X_attr4 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=350 - 50, end=350 + 50, device='cpu')
    X_attr5 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=450 - 50, end=450 + 50, device='cpu')
    X_attr6 = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=0, end=500, device='cpu')

    # "Accumulate feature importance"
    if total_attr1 is None:
        total_attr1 = X_attr1
    else:
        total_attr1 += X_attr1

    if total_attr2 is None:
        total_attr2 = X_attr2
    else:
        total_attr2 += X_attr2

    if total_attr3 is None:
        total_attr3 = X_attr3
    else:
        total_attr3 += X_attr3

    if total_attr4 is None:
        total_attr4 = X_attr4
    else:
        total_attr4 += X_attr4

    if total_attr5 is None:
        total_attr5 = X_attr5
    else:
        total_attr5 += X_attr5

    if total_attr6 is None:
        total_attr6 = X_attr6
    else:
        total_attr6 += X_attr6

# Feature importance score
avg_attr6 = abs(total_attr6) / len(test_dataset)
importance_score = torch.sum(avg_attr6, dim=1)
importance_score = importance_score.squeeze().numpy()

plt.figure(figsize=(10, 6))
plt.bar(range(len(importance_score)), importance_score)
plt.title('Importance_Score')
plt.xlabel('Index')
plt.ylabel('Importance_Score')
plt.grid(True)
plt.savefig(f'Importance_Score.svg')
plt.show()
plt.close()

# "Take the average"
avg_attr1 = abs(total_attr1) / len(test_dataset)
avg_attr2 = abs(total_attr2) / len(test_dataset)
avg_attr3 = abs(total_attr3) / len(test_dataset)
avg_attr4 = abs(total_attr4) / len(test_dataset)
avg_attr5 = abs(total_attr5) / len(test_dataset)

plt.figure(figsize=(40, 25))
custom_colors = {
    'A': 'red',
    'C': 'blue',
    'G': 'orange',
    'U': 'green'
}

# Plot the first figure
ax1 = plt.subplot(5, 1, 1)  # 5 rows, 1 column, select the 1st position
plot_logo(avg_attr1[0, :, :], ax=ax1, color=custom_colors)  # Plot feature importance logo
ax1.set_title('Midpoint of the entire sequence: index=250')

# Plot the second figure
ax2 = plt.subplot(5, 1, 2)  # 5 rows, 1 column, select the 2nd position
plot_logo(avg_attr2[0, :, :], ax=ax2, color=custom_colors)  # Plot feature importance logo
ax2.set_title('Midpoint of [0-100]: index=50')

# Plot the third figure
ax3 = plt.subplot(5, 1, 3)  # 5 rows, 1 column, select the 3rd position
plot_logo(avg_attr3[0, :, :], ax=ax3, color=custom_colors)  # Plot feature importance logo
ax3.set_title('Midpoint of [100-200]: index=150')

# Plot the fourth figure
ax4 = plt.subplot(5, 1, 4)  # 5 rows, 1 column, select the 4th position
plot_logo(avg_attr4[0, :, :], ax=ax4, color=custom_colors)  # Plot feature importance logo
ax4.set_title('Midpoint of [300-400]: index=350')

# Plot the fifth figure
ax5 = plt.subplot(5, 1, 5)  # 5 rows, 1 column, select the 5th position
plot_logo(avg_attr5[0, :, :], ax=ax5, color=custom_colors)  # Plot feature importance logo
ax5.set_title('Midpoint of [400-500]: index=450')


plt.tight_layout()
plt.savefig(f'motifs_combined.svg')
plt.show()