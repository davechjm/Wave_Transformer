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

        
    def __len__(self):
        # Adjust the total length to account for the step size
        total_steps = (len(self.data_x) - self.seq_len - self.pred_len + 1) // self.step_size
        return total_steps


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

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


class DilatedTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout_rate=0.2, dropout_=True, skip_=True):
        super(DilatedTCNBlock, self).__init__()
        
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


# Attention

class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


# %%
class DWT_MLP_Model(nn.Module):
    def __init__(self, input_channels, seq_length, pred_length ,mlp_hidden_size, output_channels, decompose_layers=3, wave='haar', mode='symmetric', nhead=8, d_model=None, num_encoder_layers=3, dropout=0.1, dilation = 2, kernel_size = 3, attention_type = 'original',  dropout_ = True, skip_ = True, general_skip_= 'skip', Revin_ = True, pos_encoder_type_ = 'Projected'):
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
        self.general_skip = general_skip_
        self.Revin = Revin_
        self.pos_encoder_type = pos_encoder_type_
        if d_model is None:
            d_model = output_channels

        
        # Assuming kernel_size can be used to derive kernel_set for illustration
        self.tcn_low = DilatedTCNBlock(input_channels, output_channels, dilation=self.dilation,kernel_size= self.kernel_size, dropout_rate= self.dropout, dropout_ = self.dropout_TF, skip_ = self.skip_TF)
        self.tcn_high_list = nn.ModuleList([DilatedTCNBlock(input_channels, output_channels, dropout_rate= self.dropout, kernel_size= self.kernel_size, dilation= self.dilation) for _ in range(decompose_layers)])

        
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

        seq_len = 24*4 #24*4*4
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
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last = False)
    #print(f"Running experiment with Data_load_type = {data_load_type}, attention_type={attention_type},dilations ={dilat}, wave_type = {wt}, mode_type = {mt},num_encoder_size = {num_encoder_size}, mlp_hidden_size = {mlp_hidden},skip_enabled={skip_enabled}, general_skip={general_skip_type}, Pos_Encoder_Type = {pos_encoder_type}, batch_size = {bs}, step_size = {s_size}, kernel_size = {k_size}, decompose_layer = {decompose_layer} ")
    dropout_rate = dr if dropout_enabled else 0.0
    # Adjust the model instantiation to include all hyperparameters
    model = DWT_MLP_Model(input_channels=321+4, seq_length=seq_, pred_length = pred_,mlp_hidden_size=mlp_hidden, 
                        output_channels=321+4, decompose_layers=decompose_layer, 
                        dropout=dropout_rate, dilation=dilat, 
                        mode=mt, wave=wt, kernel_size=k_size, 
                        attention_type=attention_type,
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

    best_model_path = f"best_model_haar_{data_load_type}_{attention_type}_{num_encoder_size}_{skip_enabled}_{general_skip_type}_{pos_encoder_type}_{bs}_{decompose_layer}_{k_size}_{s_size}_{mlp_hidden}.pt"


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
    attention_type=attention_type,
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