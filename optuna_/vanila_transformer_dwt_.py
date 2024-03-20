# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_scatter import scatter_add, scatter_max
from pytorch_wavelets import DWT1DForward, DWT1DInverse
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Optional
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch.nn import Sequential, LayerNorm, ReLU
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
import pandas as pd

from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch.utils.data import random_split

import optuna


# %%
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag, size,
                data_path,
                scale=True):
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        cols = list(df_raw.columns)
        df_raw = df_raw[cols]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [
            0,  # train
            num_train - self.seq_len,  # validation
            len(df_raw) - num_test - self.seq_len  # test
        ]
        border2s = [
            num_train,  # train
            num_train + num_vali,  # validation
            len(df_raw)  # test
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[:]
        df_data = df_raw[cols_data]

        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return torch.tensor(seq_x,  dtype=torch.float32), torch.tensor(seq_y, dtype= torch.float32)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# %%
# Specify the file path
root_path = "/home/choi/Wave_Transformer/optuna_/electricity/"
data_path = 'electricity.txt'
# Size parameters


seq_len = 96
pred_len = 96

# Instantiate datasets
train_dataset = Dataset_Custom(root_path, 'train', (seq_len, pred_len), data_path, scale=True)
val_dataset = Dataset_Custom(root_path, 'val', (seq_len, pred_len), data_path, scale=True)
test_dataset = Dataset_Custom(root_path, 'test', (seq_len, pred_len), data_path, scale=True)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last= True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle= False, drop_last = True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last = True)



# %% [markdown]
# ### Adjacency Matrix Attention

# %%
class SynthesizerCosineSimilarity(nn.Module):
    def __init__(self, in_dims, sentence_length, top_k, fixed=False):
        super(SynthesizerCosineSimilarity, self).__init__()
        self.in_dims = in_dims
        self.sentence_length = sentence_length
        self.top_k = top_k
        self.fixed = fixed

        self.value_fc = nn.Linear(in_dims, in_dims)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        value = self.value_fc(x)
        # Compute cosine similarity
        normed_x = F.normalize(x, p=2, dim=-1)
        cos_sim = torch.matmul(normed_x, normed_x.transpose(-2, -1))

        # Apply topk to make the attention matrix sparse
    
        topk_vals, topk_inds = cos_sim.topk(self.top_k, dim=-1, largest=True, sorted=True)
        attention_sparse = torch.zeros_like(cos_sim).scatter_(-1, topk_inds, topk_vals)

        # Optionally, if you want fixed attention, detach the computation
        if self.fixed:
            attention_sparse = attention_sparse.detach()

        out = torch.matmul(self.softmax(attention_sparse), value)
        return out


class TransformerBlockWithAdjacencySynthesizer(nn.Module):
    def __init__(self, in_dims, sentence_length, hidden_dim_factor, dropout,  top_k, fixed=False):
        super(TransformerBlockWithAdjacencySynthesizer, self).__init__()
        self.random_synthesizer = SynthesizerCosineSimilarity(in_dims, sentence_length, top_k,fixed)
        self.norm = nn.LayerNorm(in_dims)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_dims, in_dims * hidden_dim_factor),
            nn.ReLU(),
            nn.Linear(in_dims * hidden_dim_factor, in_dims),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_out = self.random_synthesizer(x)[0]
        x = self.dropout(self.norm(attention_out + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm(forward + x))
        return out

class UTransformerWithAdjacencySynthesizer(nn.Module):
    def __init__(self, num_features, sentence_length, hidden_dim_factor, dropout, num_layers, top_k, fixed=False):
        super(UTransformerWithAdjacencySynthesizer, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlockWithAdjacencySynthesizer(num_features, sentence_length,hidden_dim_factor, dropout, top_k, fixed) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(num_features, num_features)  # Adjust dimensions if necessary
        self.final = nn.Linear(num_features, num_features)

    def forward(self, x):
        x = self.linear(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        return x


# %%
class AdjTransformerWithDWT(nn.Module):
    def __init__(self, num_features, sentence_length,hidden_dim_factor, dropout, num_layers, wave='haar', mode='symmetric', decompose_layer=3,top_k= 15, fixed=False):
        super(AdjTransformerWithDWT, self).__init__()
        self.dwt_forward = DWT1DForward(wave=wave, J=decompose_layer, mode=mode) 
        self.dwt_inverse = DWT1DInverse(wave=wave, mode=mode)
        self.sentence_length = sentence_length
        # Adjust sentence length for the approximate transformer
        self.approx_transformer = UTransformerWithAdjacencySynthesizer(num_features, sentence_length // (2 ** decompose_layer),hidden_dim_factor, dropout, num_layers, top_k,fixed)

        # U-Net-like Transformers with Random Synthesizer for detail coefficients
        self.detail_transformers = nn.ModuleList()
        for j in range(decompose_layer):
            fixed = False
            # Adjust sentence length for each detail level
            sentence_length_adjusted = sentence_length // (2 ** (j + 1))
            self.detail_transformers.append(UTransformerWithAdjacencySynthesizer(num_features,sentence_length_adjusted, hidden_dim_factor,dropout, 1,top_k, fixed))
            

    def forward(self, x):
        x = x.permute(0, 2, 1) # Initial permutation for DWT compatibility
     
        yl, yhs = self.dwt_forward(x)
        

        # Processing yl with its dedicated Transformer and adding skip connection
        yl_transformed = self.approx_transformer(yl.permute(0, 2, 1)).permute(0, 2, 1)
        yl_combined = yl + yl_transformed  # Using '+' for skip connection

        # Process and combine each level of yhs with skip connections
        yhs_transformed = []
        for j, cD in enumerate(yhs):
            cD_transformed = self.detail_transformers[j](cD.permute(0, 2, 1)).permute(0, 2, 1)
            cD_combined = cD + cD_transformed  # Using '+' for skip connection
            yhs_transformed.append(cD_combined)

        # Reconstruct the output using inverse DWT with combined coefficients
        reconstructed = self.dwt_inverse((yl_combined, yhs_transformed))
        pred_out = reconstructed.permute(0, 2, 1)  # Final permutation

        seq_len = self.sentence_length  # Adjust based on your specific application
        pred_out = pred_out[:, -seq_len:, :]  # Slicing to match the prediction sequence length

        return pred_out


# %% [markdown]
# ### Randomized Attention

# %%
class SynthesizerRandom(nn.Module):
    def __init__(self, in_dims, sentence_length, fixed=False):
        super(SynthesizerRandom, self).__init__()
        if fixed:
            self.attention = nn.Parameter(torch.empty(1, sentence_length, sentence_length), requires_grad=False)
        else:
            self.attention = nn.Parameter(torch.empty(1, sentence_length, sentence_length), requires_grad=True)
        nn.init.xavier_uniform_(self.attention)

        self.value_fc = nn.Linear(in_dims, in_dims)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        value = self.value_fc(x)
   
        out = torch.matmul(self.softmax(self.attention), value)
        return out

class TransformerBlockWithRandomSynthesizer(nn.Module):
    def __init__(self, in_dims, sentence_length, hidden_dim_factor, dropout, fixed=False):
        super(TransformerBlockWithRandomSynthesizer, self).__init__()
        self.random_synthesizer = SynthesizerRandom(in_dims, sentence_length, fixed)
        self.norm = nn.LayerNorm(in_dims)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_dims, in_dims * hidden_dim_factor),
            nn.ReLU(),
            nn.Linear(in_dims * hidden_dim_factor, in_dims),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_out = self.random_synthesizer(x)[0]
        x = self.dropout(self.norm(attention_out + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm(forward + x))
        return out

class UTransformerWithRandomSynthesizer(nn.Module):
    def __init__(self, num_features, sentence_length, hidden_dim_factor, dropout, num_layers, fixed=False):
        super(UTransformerWithRandomSynthesizer, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlockWithRandomSynthesizer(num_features, sentence_length,hidden_dim_factor, dropout, fixed) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(num_features, num_features)  # Adjust dimensions if necessary
        self.final = nn.Linear(num_features, num_features)

    def forward(self, x):
        x = self.linear(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        return x


# %%
class UTransformerWithDWT(nn.Module):
    def __init__(self, num_features, sentence_length,hidden_dim_factor, dropout, num_layers, wave='haar', mode='symmetric', decompose_layer=3, fixed=False):
        super(UTransformerWithDWT, self).__init__()
        self.dwt_forward = DWT1DForward(wave=wave, J=decompose_layer, mode=mode) 
        self.dwt_inverse = DWT1DInverse(wave=wave, mode=mode)
        self.sentence_length = sentence_length
        # Adjust sentence length for the approximate transformer
        self.approx_transformer = UTransformerWithRandomSynthesizer(num_features, sentence_length // (2 ** decompose_layer),hidden_dim_factor, dropout, num_layers, fixed)

        # U-Net-like Transformers with Random Synthesizer for detail coefficients
        self.detail_transformers = nn.ModuleList()
        for j in range(decompose_layer):
            fixed = False
            # Adjust sentence length for each detail level
            sentence_length_adjusted = sentence_length // (2 ** (j + 1))
            self.detail_transformers.append(UTransformerWithRandomSynthesizer(num_features,sentence_length_adjusted, hidden_dim_factor,dropout, 1, fixed))
            

    def forward(self, x):
        x = x.permute(0, 2, 1) # Initial permutation for DWT compatibility
     
        yl, yhs = self.dwt_forward(x)
        

        # Processing yl with its dedicated Transformer and adding skip connection
        yl_transformed = self.approx_transformer(yl.permute(0, 2, 1)).permute(0, 2, 1)
        yl_combined = yl + yl_transformed  # Using '+' for skip connection

        # Process and combine each level of yhs with skip connections
        yhs_transformed = []
        for j, cD in enumerate(yhs):
            cD_transformed = self.detail_transformers[j](cD.permute(0, 2, 1)).permute(0, 2, 1)
            cD_combined = cD + cD_transformed  # Using '+' for skip connection
            yhs_transformed.append(cD_combined)

        # Reconstruct the output using inverse DWT with combined coefficients
        reconstructed = self.dwt_inverse((yl_combined, yhs_transformed))
        pred_out = reconstructed.permute(0, 2, 1)  # Final permutation

        seq_len = self.sentence_length  # Adjust based on your specific application
        pred_out = pred_out[:, -seq_len:, :]  # Slicing to match the prediction sequence length

        return pred_out


# %% [markdown]
# ### Vanilla Transformer

# %%
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(query, key, value)[0]
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class UTransformer(nn.Module):
    def __init__(self, num_features, embed_size, heads, forward_expansion, dropout, num_layers):
        super(UTransformer, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(num_features, embed_size)
        self.final = nn.Linear(embed_size, num_features)

    def forward(self, x):
        x = self.linear(x)
        for layer in self.layers:
            x = layer(x, x, x)
        x = self.final(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(query, key, value)[0]
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class UTransformerWithDWT(nn.Module):
    def __init__(self, num_features, embed_size, sequence_length, heads, forward_expansion, dropout, num_layers, wave='haar', mode='symmetric', decompose_layer=3):
        super(UTransformerWithDWT, self).__init__()
        self.dwt_forward = DWT1DForward(wave=wave, J=decompose_layer, mode=mode)
        self.dwt_inverse = DWT1DInverse(wave=wave, mode=mode)
        self.sequence_length = sequence_length
        # Transformer for processing yl (approximate coefficients)
        self.approx_transformer = UTransformer(num_features, embed_size, heads, forward_expansion, dropout, num_layers)

        # U-Net-like Transformers for processing yhs (detail coefficients) at varying depths
        self.detail_transformers = nn.ModuleList()
        for j in range(decompose_layer):
            #depth = num_layers - j  # Adjust the depth for each detail level
            depth = 1
            self.detail_transformers.append(UTransformer(num_features, embed_size, heads, forward_expansion, dropout, max(1, depth)))

    def forward(self, x):
        # Initial permutation for DWT compatibility
        x = x.permute(0, 2, 1)
        yl, yhs = self.dwt_forward(x)
        
        # Processing yl with its dedicated Transformer and adding skip connection
        yl_transformed = self.approx_transformer(yl.permute(0, 2, 1)).permute(0, 2, 1)
        # Skip connection for yl is achieved by combining the transformed yl with the original yl
        yl_combined = yl + yl_transformed

        # Process each level of yhs with dedicated Transformers and include skip connections
        yhs_transformed = []
        for j, cD in enumerate(yhs):
            cD_transformed = self.detail_transformers[j](cD.permute(0, 2, 1)).permute(0, 2, 1)
            # Skip connection for each level of detail coefficien
            #cD_combined = torch.cat([cD, cD_transformed], dim=2)
            cD_combined = cD  + cD_transformed
            yhs_transformed.append(cD_combined)

        # Reconstruct the output using the inverse DWT with combined coefficients
        reconstructed = self.dwt_inverse((yl_combined, yhs_transformed))

        # Final adjustments and slicing to match the prediction sequence length
        pred_out = reconstructed.permute(0, 2, 1)
        seq_len = self.sequence_length
        pred_out = pred_out[:, -seq_len:, :]

        return pred_out



# %% [markdown]
# ### Experiments with the Vanila Transformer

# %%
# Model configuration parameters
num_features = 321  # Adjust as per your dataset
embed_size = 512  # Embedding size, ensure it matches num_features or is appropriate for your model
sequence_length = 96  # Adjust based on your dataset
heads = 8
forward_expansion = 4
dropout = 0.1
num_layers =3
wave = 'haar'
mode = 'symmetric'
decompose_layer = 3
weight_decay = 0.00001
learning_rate = 0.0001

# Initialize the model
model = UTransformerWithDWT(num_features, embed_size, sequence_length, heads, forward_expansion, dropout, num_layers, wave, mode, decompose_layer)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Lists for storing loss values for visualization
train_losses = []
valid_losses = []

# Training and Validation Loop
epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for data, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for data, targets in valid_loader:
            outputs = model(data)
            loss = criterion(outputs, targets)
            valid_loss += loss.item()
    valid_loss /= len(valid_loader)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

# Testing Loop
model.eval()
test_loss = 0
with torch.no_grad():
    for data, targets in test_loader:
        outputs = model(data)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

# Visualization of Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, epochs+1), valid_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/mnt/data/training_validation_loss_plot.png')
plt.close()

# Save the train and valid losses to a text file
with open('/mnt/data/train_valid_loss.txt', 'w') as f:
    f.write('Epoch,Train Loss,Valid Loss\n')
    for epoch in range(epochs):
        f.write(f'{epoch+1},{train_losses[epoch]},{valid_losses[epoch]}\n')

# Save the test loss to a text file
with open('/mnt/data/test_loss.txt', 'w') as f:
    f.write(f'Test Loss: {test_loss:.4f}\n')
