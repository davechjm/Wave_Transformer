# %%
# %%
import time
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
from torch.nn import InstanceNorm1d
from torch import Tensor

import optuna

from RevIN import RevIN


# %%

    
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',step_size=1):
        # Initialize with seq_len and pred_len only
        if size is None:
            self.seq_len = 96  # Default value, can be adjusted
            self.pred_len = 96  # Default value, can be adjusted
        else:
            self.seq_len, self.pred_len = size[:2]
            
        
        
        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.step_size = step_size

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        cols_data = [col for col in df_raw.columns if col != 'date']
        df_data = df_raw[cols_data]
        
        num_train = int(len(df_raw) * 0.6)
        num_test = int(len(df_raw) * 0.2)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + (len(df_raw) - num_train - num_test), len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features == 'M' or self.features == 'MS':
            pass
        elif self.features == 'S':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        
        
        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]].values
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        df_stamp = df_raw[['date']].iloc[border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            # Example for time_features function
            data_stamp = self.time_features(df_stamp['date'], freq=self.freq)

        self.data_x = data[border1:border2 - self.pred_len]
        self.data_y = data[border1 + self.seq_len:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # Calculate the actual start index based on the step size
        actual_start_index = index * self.step_size

        seq_x = self.data_x[actual_start_index:actual_start_index + self.seq_len]
        seq_y = self.data_y[actual_start_index:actual_start_index + self.pred_len]
        seq_x_mark = self.data_stamp[actual_start_index:actual_start_index + self.seq_len]
        seq_y_mark = self.data_stamp[actual_start_index:actual_start_index + self.pred_len]

        return torch.tensor(seq_x,  dtype=torch.float32), torch.tensor(seq_y,  dtype=torch.float32), torch.tensor(seq_x_mark,  dtype=torch.float32), torch.tensor(seq_y_mark,  dtype=torch.float32)


    #def __len__(self):
        #return len(self.data_x) - self.seq_len + 1
        
    def __len__(self):
        # Adjust the total length to account for the step size
        total_steps = (len(self.data_x) - self.seq_len - self.pred_len + 1) // self.step_size
        return total_steps


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # Adjusted to use only seq_len and pred_len
        if size is None:
            self.seq_len = 96  # Default value for sequence length
            self.pred_len = 96  # Default value for prediction length
        else:
            self.seq_len, self.pred_len = size[0], size[2]  # Adjusted to match the input structure

        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Directly handling as multivariate
        cols_data = [col for col in df_raw.columns if col != 'date']
        df_data = df_raw[cols_data]
        
        

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Prepare timestamp data for additional time features
        tmp_stamp = df_raw[['date']].iloc[-(self.seq_len + self.pred_len):]  # Adjusted for seq_len and pred_len
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp['date'])
        pred_dates = pd.date_range(tmp_stamp['date'].values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp['date'] = list(tmp_stamp['date'].values[:-1]) + list(pred_dates[1:])  # Exclude the last current date from tmp_stamp
        if self.timeenc == 0:
            df_stamp = self.prepare_time_features(df_stamp)
        elif self.timeenc == 1:
            df_stamp = self.time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)

        # Adjustments for input and target sequences considering pred_len
        self.data_x = data[-(self.seq_len + self.pred_len):-self.pred_len]  # Input sequence
        self.data_y = data[-self.pred_len:]  # Target sequence
        self.data_stamp = df_stamp.values

    def prepare_time_features(self, df_stamp):
        df_stamp['month'] = df_stamp['date'].dt.month
        df_stamp['day'] = df_stamp['date'].dt.day
        df_stamp['weekday'] = df_stamp['date'].dt.weekday
        df_stamp['hour'] = df_stamp['date'].dt.hour
        df_stamp['minute'] = df_stamp['date'].dt.minute // 15  # Adjust if using a different frequency
        return df_stamp.drop(['date'], axis=1)

    def __getitem__(self, index):
        seq_x = self.data_x[index:index + self.seq_len]
        seq_y = self.data_y  # For prediction, seq_y serves as a placeholder
        seq_x_mark = self.data_stamp[index:index + self.seq_len]
        seq_y_mark = self.data_stamp[-self.pred_len:]

        return torch.tensor(seq_x,  dtype=torch.float32), torch.tensor(seq_y,  dtype=torch.float32), torch.tensor(seq_x_mark,  dtype=torch.float32), torch.tensor(seq_y_mark,  dtype=torch.float32)

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# %%


# %%
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.norm = InstanceNorm1d(out_channels)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))
class LogSparseMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, k_neighbors, attn_dropout=0., proj_dropout=0., qkv_bias=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.k_neighbors = k_neighbors
        self.d_k = self.d_v = d_model // n_heads

        self.W_Q = nn.Linear(d_model, self.d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads, bias=qkv_bias)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.to_out = nn.Sequential(nn.Linear(n_heads * self.d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q, K=None, V=None, key_padding_mask=None, attn_mask=None):
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Project Q, K, V
        Q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        K_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        V_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        output, attn_weights = self.sparse_attention(Q_s, K_s, V_s, key_padding_mask, attn_mask)

        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)

        return output, attn_weights

    def sparse_attention(self, q, k, v, key_padding_mask=None, attn_mask=None):
        bs, n_heads, seq_len, d_k = q.size()
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)

        # Correct handling of top-k + log-space indices
        topk_scores, topk_indices = attn_scores.topk(self.k_neighbors, dim=-1)

        # Ensure proper dimensionality for combined indices
        log_space_indices = torch.logspace(0, np.log10(seq_len-1), steps=self.k_neighbors, base=10).long()
        log_space_indices = log_space_indices[None, None, :].expand(bs, n_heads, -1)  # Expand dims to match
        combined_indices = torch.cat((topk_indices, log_space_indices), dim=-1)
        combined_indices = combined_indices.unique(sorted=True)

        # Filtering needs to adjust to correct tensor shapes
        attn_scores_filtered = torch.full_like(attn_scores, float('-inf'))
        attn_scores_filtered.scatter_(-1, combined_indices.unsqueeze(-1).expand(-1, -1, -1, seq_len), attn_scores.gather(-1, combined_indices.unsqueeze(-1).expand(-1, -1, -1, seq_len)))

        if attn_mask is not None:
            attn_scores_filtered = attn_scores_filtered.masked_fill(attn_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores_filtered, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        return output, attn_weights
class DilatedTCNBlock_original(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2,dropout_rate = 0.2):
        super(DilatedTCNBlock_original, self).__init__()
        
        # Calculate padding based on kernel size and dilation to maintain input length
        padding = (dilation * (kernel_size - 1)) // 2
        self.conv_skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)  # 1x1 conv for skip connection
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = InstanceNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        #x_skip = self.conv_skip(x)
        x = self.conv(x)
        x = self.norm(x)
        #x= x + x_skip
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
class DilatedTCNBlock_causal(nn.Module): #Causal Dilated
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2, dropout_rate=0.2):
        super(DilatedTCNBlock, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv_skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)  # 1x1 conv for skip connection
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.InstanceNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
       # x_original  = x.clone()
        # Calculate the necessary padding on the left to ensure causality
        left_padding = (self.kernel_size - 1) * self.dilation
        # Manually pad the input on the left
        x_padded = F.pad(x, (left_padding, 0))
        x = self.conv(x_padded)
        x = self.norm(x)
        # Skip connection not added in the question, but can be added if needed
        x = self.relu(x)
        x = self.dropout(x)
       # x =  x+ x_original
        return x
    
class CausalTCNBlock(nn.Module): #Non Dilated Causal
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.2):
        super(CausalTCNBlock, self).__init__()
        self.kernel_size = kernel_size
        # No dilation parameter since this is a non-dilated version
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()
        self.norm = nn.InstanceNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):        # Calculate the necessary left padding for causality
        left_padding = self.kernel_size - 1
       # x_original = x.clone()
        # Manually pad the input on the left to ensure causality
        x_padded = F.pad(x, (left_padding, 0))
        x = self.conv(x_padded)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
       # x =  x  +  x_original
        return x
        
class DilatedTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout_rate=0.2, dropout_ = True, skip_ = True):
        super(DilatedTCNBlock, self).__init__()
        
        # Calculate padding based on kernel size and dilation to maintain input length
        padding = (dilation * (kernel_size - 1)) // 2
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = InstanceNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_ = dropout_
        self.skip_ = skip_

    def forward(self, x):
        x_original = x.clone()
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        if self.dropout_:
            x = self.dropout(x)
        else:
            x = x
        if self.skip_:
            x = x + x_original
        else:
            x = x
        return x      
        
class DilatedTCNBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout_rate=0.2, dropout_=True, skip_=True):
        super(DilatedTCNBlock2, self).__init__()
        
        # Calculate padding based on kernel size and dilation to maintain input length
        padding = (dilation * (kernel_size - 1)) // 2
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = InstanceNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_ else nn.Identity()

        # Second convolutional layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = InstanceNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_ else nn.Identity()

        self.skip_ = skip_

    def forward(self, x):
        x_original = x.clone()
        
        # First conv -> norm -> relu -> dropout
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Second conv -> norm -> relu -> dropout
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        if self.skip_:
            x = x + x_original
        
        return x



# %%
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model%2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    


class ProjectedPositionalEncoding(nn.Module):
    def __init__(self, d_input, d_model, dropout=0.1, max_len=5000):
        super(ProjectedPositionalEncoding, self).__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Projection layer: maps input dimension to model dimension
        self.projection = nn.Linear(d_input, d_model)

        self.positionalencoding = PositionalEncoding(d_model,dropout, max_len = 5000)

    def forward(self, x):
        # Apply linear projection
        x_projected = self.projection(x)
        # Add positional encoding
        x_pe = x_projected + self.positionalencoding(x)
        return x_pe


# %%
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model,d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x, attn = self.attention(
            x, x, x
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x

# %%
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    p_attn = torch.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

def log_sparse_attention(query, key, value, k=10, mask=None, dropout=None):
    batch_size, seq_length, d_k = query.size()
    k =  seq_length//2
    # Create a sparse mask
    sparse_mask = generate_log_sparse_plus_neighbors_mask(seq_length, k).to(query.device)
    if mask is not None:
        # Combine the existing mask with the sparse mask
        mask = mask.logical_and(sparse_mask)
    else:
        mask = sparse_mask

    # Compute scaled dot-product attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply the combined mask to the scores
    scores = scores.masked_fill(mask == 0, float('-inf'))

    # Compute attention probabilities
    p_attn = torch.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    # Return the weighted sum of the values
    return torch.matmul(p_attn, value), p_attn

def generate_log_sparse_mask(seq_length):
    """
    Generates a log-sparse mask for the attention mechanism.
    This is a simple pattern-based approach where each position
    attends to its previous position and other positions based on
    a log step pattern.
    """
    mask = torch.zeros(seq_length, seq_length)
    for i in range(seq_length):
        mask[i, i] = 1  # Attend to self
        # Log step pattern
        step = 1
        while i - step >= 0:
            mask[i, i - step] = 1
            step *= 2
    return mask.bool()

def generate_log_sparse_plus_neighbors_mask(seq_length, k):
    """
    Generates a mask for the attention mechanism that combines log-sparse attention with
    attention to k immediate neighbors. Each position attends to itself, its k previous neighbors,
    and other positions based on a log step pattern.
    
    :param seq_length: The length of the sequence.
    :param k: The number of immediate neighbors to attend to.
    :return: A boolean mask indicating allowed attentions.
    """
    mask = torch.zeros(seq_length, seq_length)
    for i in range(seq_length):
        mask[i, i] = 1  # Attend to self
        # Immediate k neighbors
        for j in range(max(0, i-k), i):
            mask[i, j] = 1
        # Log step pattern
        step = 1
        while i - step >= 0:
            mask[i, i - step] = 1
            step *= 2
    return mask.bool()





# %%
class DWT_MLP_Model(nn.Module):
    def __init__(self, input_channels, seq_length, pred_length ,mlp_hidden_size, output_channels, decompose_layers=3, wave='haar', mode='symmetric', nhead=8, d_model=None, num_encoder_layers=3, dropout=0.1, dilation = 2, kernel_size = 3, attention_type = 'original', TCN_type = 'dilated', dropout_ = True, skip_ = True, general_skip_= 'skip', Revin_ = True, pos_encoder_type_ = 'Projected'):
        super(DWT_MLP_Model, self).__init__()
        self.dwt_forward = DWT1DForward(J=decompose_layers, wave=wave, mode=mode)
        self.dwt_inverse = DWT1DInverse(wave=wave, mode=mode)
        self.seq_len = seq_length
        self.pred_len = pred_length
        self.dropout = dropout
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.num_decoder_layers = num_encoder_layers
        self.dropout_TF = dropout_
        self.skip_TF = skip_
        self.attention_type = attention_type
        self.TCN_type = TCN_type
        self.general_skip = general_skip_
        self.Revin = Revin_
        self.pos_encoder_type = pos_encoder_type_
        if d_model is None:
            d_model = output_channels

        # Convolutional and TCN blocks for low and high-frequency components
        self.conv_low = ConvBlock(input_channels, output_channels)
        if self.TCN_type == 'dilated':
            self.tcn_low = DilatedTCNBlock(input_channels, output_channels, dilation=self.dilation,kernel_size= self.kernel_size, dropout_rate= self.dropout, dropout_ = self.dropout_TF, skip_ = self.skip_TF) 
            self.tcn_high_list = nn.ModuleList([DilatedTCNBlock(input_channels, output_channels, dropout_rate= self.dropout, kernel_size= self.kernel_size, dilation= self.dilation) for _ in range(decompose_layers)])
        else:
            # Assuming kernel_size can be used to derive kernel_set for illustration
            self.tcn_low = DilatedTCNBlock2(input_channels, output_channels, dilation=self.dilation,kernel_size= self.kernel_size, dropout_rate= self.dropout, dropout_ = self.dropout_TF, skip_ = self.skip_TF)
            self.tcn_high_list = nn.ModuleList([DilatedTCNBlock2(input_channels, output_channels, dropout_rate= self.dropout, kernel_size= self.kernel_size, dilation= self.dilation) for _ in range(decompose_layers)])

        
        if self.Revin:
            self.revin_layer = RevIN(input_channels)
        
        # Positional Encoding
        if  self.pos_encoder_type == 'Projected':  
            self.pos_encoder = ProjectedPositionalEncoding(input_channels, d_model, max_len=5000)
        else:
            self.pos_encoder = PositionalEncoding(d_model, max_len = 5000)
        # Custom Transformer Encoder setup
        if self.attention_type == 'original':
            self.attention = attention  # Assuming attention is defined elsewhere
        else:
            self.attention = log_sparse_attention
        encoder_layers_low = [EncoderLayer(self.attention, d_model, d_ff=mlp_hidden_size, dropout=self.dropout, activation="relu") for _ in range(num_encoder_layers)]
        self.transformer_low = Encoder(encoder_layers_low, norm_layer=nn.LayerNorm(d_model))

        # Transformer Encoders for high-frequency components, using custom Encoder
        self.transformer_high_list = nn.ModuleList(
            [Encoder([EncoderLayer(self.attention, d_model, d_ff=mlp_hidden_size, dropout=self.dropout, activation="relu") for _ in range(num_encoder_layers)]) 
             for _ in range(decompose_layers)])
        

    def forward(self, x):
        #x = x.permute(0, 2, 1)  # Adjust dimensions for DWT
        if self.Revin:
            x = self.revin_layer(x, 'norm')
            x = x.permute(0,2,1)
        else:
            x = x.permute(0,2,1)
        
            
        x_low, x_highs = self.dwt_forward(x)
        x_low_tcn = self.tcn_low(x_low)
        x_low_combined = x_low_tcn
        
        if self.TCN_type == 'dilated':
            if self.general_skip == 'skip':
                 x_low_combined = x_low + x_low_combined
            else:
                x_low_combined = x_low_combined
                
        else:
            if self.general_skip == 'skip':
                x_low_combined = x_low + x_low_combined
            else:
                x_low_combined = x_low_combined
                
        x_low_combined = x_low_combined.permute(0,2,1)
    
        x_low_combined = self.pos_encoder(x_low_combined)
        x_low_combined, _ = self.transformer_low(x_low_combined) # Adjusted for custom encoder
        x_low_combined = x_low_combined.permute(0,2,1)
        x_low_combined = x_low + x_low_combined
        
        # Process high-frequency components
        x_highs_processed = []
        for i, x_high in enumerate(x_highs):
            x_high_tcn = self.tcn_high_list[i](x_high)
            x_high_combined = x_high_tcn
            if self.TCN_type == 'dilated':
                if self.general_skip == 'skip':
                    x_high_combined = x_high + x_high_combined
                else:
                    x_high_combined = x_high_combined
            else:
                if self.general_skip == 'skip':
                    
                    x_high_combined = x_high + x_high_combined
                else:
                    x_high_combined = x_high_combined
            x_high_combined = x_high_combined.permute(0,2,1)
         
            x_high_combined = self.pos_encoder(x_high_combined)
            x_high_combined, _ = self.transformer_high_list[i](x_high_combined)  # Adjusted for custom encoder
            x_high_combined = x_high.permute(0,2,1) + x_high_combined
            x_highs_processed.append(x_high_combined.permute(0,2,1))
        
        # Reconstruct the signal and adjust dimensions
        pred_out = self.dwt_inverse((x_low_combined, x_highs_processed)).permute(0, 2, 1)
        if self.Revin:
            pred_out = self.revin_layer(pred_out, 'denorm')
        else:
            pred_out = pred_out
        
        pred_out = pred_out[:, :, :-4] # Do not make predictions for meta features
        pred_out = pred_out[:, -self.pred_len:, :]


        return pred_out




# Assuming DWT_MLP_Model is defined elsewhere, along with the necessary imports
seq_ = 24*4*4
pred_ = 24*4
# Define hyperparameter combinations
dropout_enabled = True
skip_enabled = True
revin_type  = True
data_load_type = 'multivariate'
TCN_type = 'dilated2'
attention_type = 'original'
num_encoder_size = 1
pos_encoder_type = 'Projected'
general_skip_type = 'skip'
mlp_hidden = 128
k_size = 5
s_size = 8
decompose_layer = 1
bs = 64
mt = 'zero'
wt = 'haar'
dilat = 3
# Define the ranges for the hyperparameters
learning_rates = np.logspace(-3, -2, 100)  # Learning rates between 1e-3 and 1e-2
dropout_rates = np.linspace(0.0, 0.2, 100)  # Dropout rates between 0 and 0.5
weight_decays = np.logspace(-4, -3, 100)  # Weight decays between 1e-4 and 1e-3
indices = np.random.choice(range(100), size=100, replace=False)


count = 0

for i in indices:
    lrs = learning_rates[i]
    dr = dropout_rates[i]
    wd = weight_decays[i]
    #lrs =0.0052230056036904522
    #dr = 0.10146011891748014
    #wd = 1.0059977697794999e-04
                                            
    if data_load_type == 'univariate':

        # Define the root path where your dataset is located and the name of your dataset file
        root_path = '/home/choi/Wave_Transformer/optuna_/electricity/'
        data_path = 'electricity.csv'

        # Define the size configuration for your dataset
        seq_len = 24 * 4 *4    # Length of input sequences
        label_len = 24 * 4      # Length of labels within the sequence to predict
        pred_len = 24 * 4       # Number of steps to predict into the future

        size = [seq_len, label_len, pred_len]

        # Initialize the custom dataset for training, validation, and testing
        train_dataset = Dataset_Custom(root_path=root_path, features= 'S', flag='train',  data_path=data_path, step_size = s_size)
        val_dataset = Dataset_Custom(root_path=root_path, features= 'S',flag='val', data_path=data_path,step_size = s_size)
        test_dataset = Dataset_Custom(root_path=root_path, features= 'S',flag='test',  data_path=data_path,step_size = s_size)

        # Optionally, initialize the dataset for prediction (if needed)
        #pred_dataset = Dataset_Pred(root_path=root_path, flag='pred', size=size, data_path=data_path, inverse=True)

        # Example on how to create DataLoaders for PyTorch training (adjust batch_size as needed)
        batch_size = bs
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last = True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last = False)
        
    else:
        
        # Specify the file path
        root_path = '/home/choi/Wave_Transformer/optuna_/electricity/'
        data_path = 'electricity.csv'
        # Size parameters

        seq_len = 24*4*4
        pred_len = 24*4
        #batch_size = bs
        # Initialize the custom dataset for training, validation, and testing
        train_dataset = Dataset_Custom(root_path=root_path, features= 'M', flag='train', data_path=data_path, step_size =s_size)
        val_dataset = Dataset_Custom(root_path=root_path, features= 'M',flag='val', data_path=data_path,step_size = s_size)
        test_dataset = Dataset_Custom(root_path=root_path, features= 'M',flag='test', data_path=data_path,step_size = s_size)

        # Optionally, initialize the dataset for prediction (if needed)
        #pred_dataset = Dataset_Pred(root_path=root_path, flag='pred', size=size, data_path=data_path, inverse=True)

        # Example on how to create DataLoaders for PyTorch training (adjust batch_size as needed)
        batch_size = bs
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last = True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last = True)
    #print(f"Running experiment with Data_load_type = {data_load_type}, TCN_type={TCN_type}, attention_type={attention_type},dilations ={dilat}, wave_type = {wt}, mode_type = {mt},num_encoder_size = {num_encoder_size}, mlp_hidden_size = {mlp_hidden},skip_enabled={skip_enabled}, general_skip={general_skip_type}, Pos_Encoder_Type = {pos_encoder_type}, batch_size = {bs}, step_size = {s_size}, kernel_size = {k_size}, decompose_layer = {decompose_layer} ")
    dropout_rate = dr if dropout_enabled else 0.0
    # Adjust the model instantiation to include all hyperparameters
    model = DWT_MLP_Model(input_channels=321+4, seq_length=seq_, pred_length = pred_,mlp_hidden_size=mlp_hidden, 
                        output_channels=321+4, decompose_layers=decompose_layer, 
                        dropout=dropout_rate, dilation=dilat, 
                        mode=mt, wave=wt, kernel_size=k_size, 
                        attention_type=attention_type, TCN_type=TCN_type, 
                        num_encoder_layers=num_encoder_size, nhead=8, 
                        dropout_=dropout_enabled,
                        skip_=True, general_skip_=general_skip_type, Revin_=revin_type, pos_encoder_type_ = pos_encoder_type)

    # Define criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lrs, 
                        weight_decay=wd)

    train_losses = []
    val_losses = []

    # Early stopping parameters
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0

    num_epochs = 30

    # Start the timer
    start_time = time.time()

    best_model_path = f"best_model_haar_{data_load_type}_{TCN_type}_{attention_type}_{num_encoder_size}_{skip_enabled}_{general_skip_type}_{pos_encoder_type}_{bs}_{decompose_layer}_{k_size}_{s_size}_{mlp_hidden}.pt"


    for epoch in range(num_epochs):
        model.train()

        train_loss = 0.0

        for seq_x, seq_y, seq_x_mark, seq_y_mark in train_loader:

            inputs = torch.cat((seq_x, seq_x_mark), dim=-1)
        
            targets = torch.cat((seq_y, seq_y_mark), dim=-1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets[:, :, :-4])  # Assuming specific output handling
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_x, seq_y, seq_x_mark, seq_y_mark in val_loader:
                inputs = torch.cat((seq_x, seq_x_mark), dim=-1)
                targets = torch.cat((seq_y, seq_y_mark), dim=-1)
                outputs = model(inputs)
                loss = criterion(outputs, targets[:, :, :-4])  # Same output handling assumption
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            #print(f"New best model saved at {best_model_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total Model Running Time: {total_time:.2f} seconds')
    best_model = DWT_MLP_Model(input_channels=321+4, seq_length=seq_, pred_length = pred_, mlp_hidden_size=mlp_hidden, 
    output_channels=321+4, decompose_layers=decompose_layer, dropout=dropout_rate, dilation=dilat, 
    mode=mt, wave=wt, kernel_size=k_size, 
    attention_type=attention_type, TCN_type=TCN_type, 
    num_encoder_layers=num_encoder_size, nhead=8, 
    dropout_=dropout_enabled,
    skip_=True, general_skip_=general_skip_type, Revin_=revin_type, pos_encoder_type_ = pos_encoder_type)
    best_model.load_state_dict(torch.load(best_model_path))
    # Evaluation on test data
    best_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in test_loader:
            inputs = torch.cat((seq_x, seq_x_mark), dim=-1)
            targets = torch.cat((seq_y, seq_y_mark), dim=-1)
            outputs = best_model(inputs)
            loss = criterion(outputs, targets[:, :, :-4])  # Assuming specific output handling
            test_loss += loss.item() * inputs.size(0)

    test_loss /= len(test_loader.dataset)
    print(f'The {count}th model done.')
    count += 1
    print(f'Test Loss for configuration: , learning_rate = {lrs}, weight_decay = {wd}, dropout_rate = {dr}: {test_loss:.4f}')





# %%



