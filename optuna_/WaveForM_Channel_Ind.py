from __future__ import division
import time
import torch
print("PyTorch version:", torch.__version__)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_scatter import scatter_add, scatter_max
from pytorch_wavelets import DWT1DForward, DWT1DInverse
import warnings

import numbers

from torch.nn import Parameter

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
from scipy.stats import pearsonr

from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch.utils.data import random_split
from torch.nn import InstanceNorm1d
from torch import Tensor
from torch.autograd import Variable
import torch.nn.init as init

import optuna

from RevIN import RevIN

def load_dataset(root_path, data_path, drop_columns='date'):
    """
    Load dataset from a CSV file, drop specified columns, and apply standard scaling.

    Args:
    root_path (str): Directory path where the data file is located.
    data_path (str): Name of the data file.
    drop_columns (list of str): Columns to be dropped from the dataset.

    Returns:
    numpy.ndarray: Scaled dataset as a NumPy array.
    """
    # Combine the root path and the data path
    full_path = os.path.join(root_path, data_path)
    
    # Read the data from CSV file
    data = pd.read_csv(full_path)
    
    # Drop specified columns if provided
    if drop_columns:
        data.drop(columns=drop_columns, inplace=True, errors='ignore')
    
    # Apply StandardScaler to normalize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)  # Assuming data to be normalized is numeric

    return scaled_data

    
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



class Linear(nn.Module):
    def __init__(self, c_in: int, c_out: int, bias: bool = True):
        super(Linear, self).__init__()
        self._mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        return self._mlp(X)


class MixProp(nn.Module):
    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super(MixProp, self).__init__()
        self._mlp = Linear((gdep + 1) * c_in, c_out)
        self._gdep = gdep
        self._dropout = dropout
        self._alpha = alpha

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor, A: torch.FloatTensor) -> torch.FloatTensor:
        A = A + torch.eye(A.size(0)).to(X.device)
        d = A.sum(1)
        H = X
        H_0 = X
        A = A / d.view(-1, 1)
        for _ in range(self._gdep):
            
            H = self._alpha * X + (1 - self._alpha) * torch.einsum(
                "ncwl,vw->ncvl", (H, A)
            )
            H_0 = torch.cat((H_0, H), dim=1)
        H_0 = self._mlp(H_0)
        return H_0


class DilatedInception(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_set: list, dilation_factor: int):
        super(DilatedInception, self).__init__()
        self._time_conv = nn.ModuleList()
        self._kernel_set = kernel_set
        c_out = int(c_out / len(self._kernel_set))
        for kern in self._kernel_set:
            self._time_conv.append(
                nn.Conv2d(c_in, c_out, (1, kern), dilation=(1, dilation_factor))
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X_in: torch.FloatTensor) -> torch.FloatTensor:
        
        X = []
        for i in range(len(self._kernel_set)):
            X.append(self._time_conv[i](X_in))
        
        for i in range(len(self._kernel_set)):
            X[i] = X[i][..., -X[-1].size(3) :]
        
        
        Y = [0 ,0 ,0, 0]
        for i in range(len(self._kernel_set)):
            Y[i] = X[i].permute(0, 2, 1, 3)
        
        Y = torch.cat(Y, dim=2)
        
        X = Y.permute(0, 2, 1, 3)
        return  X



class LayerNormalization(nn.Module):
    __constants__ = ["normalized_shape", "weight", "bias", "eps", "elementwise_affine"]

    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super(LayerNormalization, self).__init__()
        self._normalized_shape = tuple(normalized_shape)
        self._eps = eps
        self._elementwise_affine = elementwise_affine
        if self._elementwise_affine:
            self._weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self._bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("_weight", None)
            self.register_parameter("_bias", None)
        self._reset_parameters()

    def _reset_parameters(self):
        if self._elementwise_affine:
            init.ones_(self._weight)
            init.zeros_(self._bias)

    def forward(self, X: torch.FloatTensor, idx: torch.LongTensor) -> torch.FloatTensor:
        if self._elementwise_affine:
            return F.layer_norm(
                X,
                tuple(X.shape[1:]),
                self._weight[:, idx, :],
                self._bias[:, idx, :],
                self._eps,
            )
        else:
            return F.layer_norm(
                X, tuple(X.shape[1:]), self._weight, self._bias, self._eps
            )


class GPModuleLayer(nn.Module):

    def __init__(
        self,
        dilation_exponential: int,
        rf_size_i: int,
        kernel_size: int,
        j: int,
        residual_channels: int,
        conv_channels: int,
        skip_channels: int,
        kernel_set: list,
        new_dilation: int,
        layer_norm_affline: bool,
        gcn_true: bool,
        seq_length: int,
        receptive_field: int,
        dropout: float,
        gcn_depth: int,
        num_nodes: int,
        propalpha: float,
    ):
        super(GPModuleLayer, self).__init__()
        self._dropout = dropout
        self._gcn_true = gcn_true
        
        if dilation_exponential > 1:
            rf_size_j = int(
                rf_size_i
                + (kernel_size - 1)
                * (dilation_exponential ** j - 1)
                / (dilation_exponential - 1)
            )
        else:
            rf_size_j = rf_size_i + j * (kernel_size - 1)
        
        self._filter_conv = DilatedInception(
            residual_channels,
            conv_channels,
            kernel_set=kernel_set,
            dilation_factor=new_dilation,
        )
        
        self._gate_conv = DilatedInception(
            residual_channels,
            conv_channels,
            kernel_set=kernel_set,
            dilation_factor=new_dilation,
        )

        self._residual_conv = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )

        if seq_length > receptive_field:
            self._skip_conv = nn.Conv2d(
                in_channels=conv_channels,
                out_channels=skip_channels,
                kernel_size=(1, seq_length - rf_size_j + 1),
            )
        else:
            self._skip_conv = nn.Conv2d(
                in_channels=conv_channels,
                out_channels=skip_channels,
                kernel_size=(1, receptive_field - rf_size_j + 1),
            )

        if gcn_true:
            self._mixprop_conv1 = MixProp(
                conv_channels, residual_channels, gcn_depth, dropout, propalpha
            )

            self._mixprop_conv2 = MixProp(
                conv_channels, residual_channels, gcn_depth, dropout, propalpha
            )

        if seq_length > receptive_field:
            self._normalization = LayerNormalization(
                (residual_channels, num_nodes, seq_length - rf_size_j + 1),
                elementwise_affine=layer_norm_affline,
            )

        else:
            self._normalization = LayerNormalization(
                (residual_channels, num_nodes, receptive_field - rf_size_j + 1),
                elementwise_affine=layer_norm_affline,
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self,
        X: torch.FloatTensor,
        X_skip: torch.FloatTensor,
        A_tilde: Optional[torch.FloatTensor],
        idx: torch.LongTensor,
        training: bool,
    ) -> torch.FloatTensor:
        
        X_residual = X
        X_filter = self._filter_conv(X)
        X_filter = torch.tanh(X_filter)
        X_gate = self._gate_conv(X)
        X_gate = torch.sigmoid(X_gate)
        X = X_filter * X_gate
        X = F.dropout(X, self._dropout, training=training)
        X_skip = self._skip_conv(X) + X_skip
        if self._gcn_true:
            X = self._mixprop_conv1(X, A_tilde) + self._mixprop_conv2(
                X, A_tilde.transpose(1, 0)
            )
        else:
            X = self._residual_conv(X)

        X = X + X_residual[:, :, :, -X.size(3) :]
        X = self._normalization(X, idx)
        return X, X_skip


class GPModule(nn.Module):
    
    def __init__(
        self,
        gcn_true: bool,
        build_adj: bool,
        gcn_depth: int,
        num_nodes: int,
        kernel_set: list,
        kernel_size: int,
        dropout: float,
        dilation_exponential: int,
        conv_channels: int,
        residual_channels: int,
        skip_channels: int,
        end_channels: int,
        seq_length: int,
        in_dim: int,
        out_dim: int,
        layers: int,
        propalpha: float,
        layer_norm_affline: bool,
        graph_constructor,
        xd: Optional[int] = None,
    ):
        super(GPModule, self).__init__()
        
        self._gcn_true = gcn_true
        self._build_adj_true = build_adj
        self._num_nodes = num_nodes
        self._dropout = dropout
        self._seq_length = seq_length
        self._layers = layers
        self._idx = torch.arange(self._num_nodes)
        
        self._gp_layers = nn.ModuleList()
        
        self._graph_constructor = graph_constructor
        
        self._set_receptive_field(dilation_exponential, kernel_size, layers)
        
        new_dilation = 1
        for j in range(1, layers + 1):
            self._gp_layers.append(
                GPModuleLayer(
                    dilation_exponential=dilation_exponential,
                    rf_size_i=1,
                    kernel_size=kernel_size,
                    j=j,
                    residual_channels=residual_channels,
                    conv_channels=conv_channels,
                    skip_channels=skip_channels,
                    kernel_set=kernel_set,
                    new_dilation=new_dilation,
                    layer_norm_affline=layer_norm_affline,
                    gcn_true=gcn_true,
                    seq_length=seq_length,
                    receptive_field=self._receptive_field,
                    dropout=dropout,
                    gcn_depth=gcn_depth,
                    num_nodes=num_nodes,
                    propalpha=propalpha,
                )
            )
            
            new_dilation *= dilation_exponential
        
        self._setup_conv(
            in_dim, skip_channels, end_channels, residual_channels, out_dim
        )
        
        self._reset_parameters()
        
    def _setup_conv(
        self, in_dim, skip_channels, end_channels, residual_channels, out_dim
    ):
    
        self._start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )
        
        if self._seq_length > self._receptive_field:
            
            self._skip_conv_0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self._seq_length),
                bias=True,
            )
            
            self._skip_conv_E = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, self._seq_length - self._receptive_field + 1),
                bias=True,
            )
            
        else:
            self._skip_conv_0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self._receptive_field),
                bias=True,
            )
            
            self._skip_conv_E = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, 1),
                bias=True,
            )
        
        self._end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        
        self._end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def _set_receptive_field(self, dilation_exponential, kernel_size, layers):
        if dilation_exponential > 1:
            self._receptive_field = int(
                1
                + (kernel_size - 1)
                * (dilation_exponential ** layers - 1)
                / (dilation_exponential - 1)
            )
        else:
            self._receptive_field = layers * (kernel_size - 1) + 1
    
    def forward(
        self,
        context: torch.FloatTensor,
        A_tilde: Optional[torch.FloatTensor] = None,
        idx: Optional[torch.LongTensor] = None,
        FE: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        
        X_in = context.permute(0, 3, 2, 1)
        
        seq_len = X_in.size(3)
        assert (
            seq_len == self._seq_length
        ), "Input sequence length not equal to preset sequence length."
        
        if self._seq_length < self._receptive_field:
            X_in = nn.functional.pad(
                X_in, (self._receptive_field - self._seq_length, 0, 0, 0)
            )
        
        if self._gcn_true:
            if self._build_adj_true:
                if idx is None:
                    A_tilde = self._graph_constructor(self._idx.to(X_in.device), FE=FE)
                else:
                    A_tilde = self._graph_constructor(idx, FE=FE)
        
        X = self._start_conv(X_in)
        X_skip = self._skip_conv_0(
            F.dropout(X_in, self._dropout, training=self.training)
        )
        if idx is None:
            for gp in self._gp_layers:
                
                X, X_skip = gp(X, X_skip, A_tilde, self._idx.to(X_in.device), self.training)
        else:
            for gp in self._gp_layers:
                X, X_skip = gp(X, X_skip, A_tilde, idx, self.training)
        
        X_skip = self._skip_conv_E(X) + X_skip
        X = F.relu(X_skip)
        X = F.relu(self._end_conv_1(X))
        X = self._end_conv_2(X)
        
        return X
    
    
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(name)
                print(param.shape)


class GraphConstructor(nn.Module):
    def __init__(
        self, nnodes: int, k: int, dim: int, alpha: float, xd: Optional[int] = None
    ):
        super(GraphConstructor, self).__init__()
        if xd is not None:
            self._static_feature_dim = xd
            self._linear1 = nn.Linear(xd, dim)
            self._linear2 = nn.Linear(xd, dim)
        else:
            self._embedding1 = nn.Embedding(nnodes, dim)
            self._embedding2 = nn.Embedding(nnodes, dim)
            self._linear1 = nn.Linear(dim, dim)
            self._linear2 = nn.Linear(dim, dim)
        
        self._k = k
        self._alpha = alpha
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self, idx: torch.LongTensor, FE: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        
        if FE is None:
            nodevec1 = self._embedding1(idx)
            nodevec2 = self._embedding2(idx)
        else:
            assert FE.shape[1] == self._static_feature_dim
            nodevec1 = FE[idx, :]
            nodevec2 = nodevec1 
        
        nodevec1 = torch.tanh(self._alpha * self._linear1(nodevec1))
        nodevec2 = torch.tanh(self._alpha * self._linear2(nodevec2))
    
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        A = F.relu(torch.tanh(self._alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(A.device)
        mask.fill_(float("0"))
        
        s1, t1 = A.topk(self._k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        A = A * mask
        return A

class Model_(nn.Module):
    def __init__(self, seq_len, pred_len, n_points, dropout, wavelet_j, wavelet, subgraph_size, node_dim, n_gnn_layer):
        super(Model_, self).__init__()
        self.seq_len = seq_len # Sequence Length
        self.pred_len = pred_len # Pred Length
        self.points = n_points # Num Features
        self.dropout = dropout #dropout rate
        
        decompose_layer = wavelet_j # the number of wavelet decompose layer
        wave = wavelet # wavelet function e.g.) 'haar'
        
        mode = 'symmetric' 
        self.dwt = DWT1DForward(wave=wave, J=decompose_layer, mode=mode)  
        self.idwt = DWT1DInverse(wave=wave)
        
        
        tmp1 = torch.randn(1, 1, self.seq_len)
        tmp1_yl, tmp1_yh = self.dwt(tmp1)
        tmp1_coefs = [tmp1_yl] + tmp1_yh
        
        tmp2 = torch.randn(1, 1, self.seq_len + self.pred_len)
        tmp2_yl, tmp2_yh = self.dwt(tmp2)
        tmp2_coefs = [tmp2_yl] + tmp2_yh
        assert decompose_layer + 1 == len(tmp1_coefs) == len(tmp2_coefs)
        
        self._graph_constructor = GraphConstructor(
            nnodes=self.points,
            k=subgraph_size, # topk
            dim=node_dim, # node_dim in graph
            alpha=3.0
        )
        
        
        self.nets = nn.ModuleList()
        for i in range(decompose_layer + 1):
            self.nets.append(
                GPModule(
                    gcn_true = True,
                    build_adj = True,
                    gcn_depth=2,
                    num_nodes=self.points,
                    kernel_set=[2, 3, 6, 7],
                    kernel_size=7,
                    dropout=self.dropout,
                    conv_channels=32,
                    residual_channels=32,
                    skip_channels=64,
                    end_channels=128,
                    seq_length=(tmp1_coefs[i].shape[-1]),
                    in_dim=1,
                    out_dim=(tmp2_coefs[i].shape[-1]) - (tmp1_coefs[i].shape[-1]),
                    layers=n_gnn_layer, # the nubmer of layers of gnn
                    propalpha=0.05,
                    dilation_exponential=2,
                    graph_constructor=self._graph_constructor,
                    layer_norm_affline=True,
                )
            )
    
    
    def model(self, coefs):
        new_coefs = []
        for coef, net in zip(coefs, self.nets):
            new_coef = net(coef.permute(0,2,1).unsqueeze(-1))
            new_coefs.append(new_coef.squeeze().permute(0,2,1))
        
        return new_coefs
        
    
    
    def forward(self, x_enc):
        in_dwt = x_enc.permute(0,2,1)
        
        yl, yhs = self.dwt(in_dwt)
        coefs = [yl] + yhs
        
        
        coefs_new = self.model(coefs)
        
        coefs_idwt = []
        for i in range(len(coefs_new)):
            coefs_idwt.append(torch.cat((coefs[i], coefs_new[i]), 2))
        
        
        out = self.idwt((coefs_idwt[0], coefs_idwt[1:]))
        pred_out = out.permute(0, 2, 1)
        
        
        return pred_out[:, -self.pred_len:, :]

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# Define a function to extract correlated features for each feature
def extract_correlated_features(autocorrelation_matrix, num_correlated_features=10):
    num_features = autocorrelation_matrix.shape[0]
    correlated_features = {}
    for i in range(num_features):
        # Find indices of the top correlated features
        correlated_indices = np.argsort(autocorrelation_matrix[i])[::-1][:num_correlated_features]
        correlated_features[i] = correlated_indices
    return correlated_features   

def compute_autocorrelation(seq_x):
    num_features = seq_x.shape[1]  # seq_x is now [seq_len, num_channels]
    autocorrelation_matrix = np.zeros((num_features, num_features))
    
    for i in range(num_features):
        for j in range(num_features):
            if i == j:
                autocorrelation_matrix[i, j] = 1.0  # Auto-correlation is always 1
            else:
                # Compute the absolute value of Pearson correlation coefficient
                autocorrelation_matrix[i, j] = np.abs(pearsonr(seq_x[:, i], seq_x[:, j])[0])
    
    return autocorrelation_matrix

class Model(nn.Module):
    def __init__(self, seq_len, pred_len, n_points, dropout, wavelet_j, wavelet, subgraph_size, node_dim, n_gnn_layer, correlated_groups):
        super(Model, self).__init__()
        self.channels = n_points
        self.correlated_groups = correlated_groups  # Pass the correlated groups as a parameter
        self.backbone_modules = nn.ModuleDict({
            str(i): Model_(
                seq_len=seq_len,
                pred_len=pred_len,
                n_points=len(group),
                dropout=dropout,
                wavelet_j=wavelet_j,
                wavelet=wavelet,
                subgraph_size=subgraph_size,
                node_dim=node_dim,
                n_gnn_layer=n_gnn_layer
            )
            for i, group in enumerate(correlated_groups.values())
        })
        self.pred_len = pred_len

    def forward(self, x):
        x_ = x.clone()
        output = torch.zeros([x_.size(0), self.pred_len, x_.size(2)], dtype=x_.dtype)
        # Dictionary to collect predictions for averaging
        predictions = {i: [] for i in range(x_.size(2))}
        
        for idx, group in enumerate(self.correlated_groups.values()):
            # Extracting the specific features for this group
            # Convert group to a tensor properly, ensuring it's a contiguous copy if it's a NumPy array
            if isinstance(group, np.ndarray):
                group_tensor = torch.tensor(group.copy(), dtype=torch.long, device=x_.device)
            else:
                group_tensor = torch.tensor(group, dtype=torch.long, device=x_.device)

      
            group = np.sort(group).copy()
            group_tensor = torch.tensor(group, dtype=torch.long, device=x_.device)
            try:
                group_data = x_[:, :, group_tensor]
            except Exception as e:
                print("Failed during indexing:")
                print("Group tensor:", group_tensor)
                print("Original indices:", group)
                print("Tensor shape:", x.shape)
                print("Tensor stride:", x.stride())
                print("Is contiguous:", x.is_contiguous())
                raise e  # Re-raise the exception to see the traceback with added context

            
            pred = self.backbone_modules[str(idx)](group_data)
            for i, feature_index in enumerate(group):
                predictions[feature_index].append(pred[:, :, i:i+1])

        # Averaging predictions for each feature
        for i in range(x_.size(2)):
            if predictions[i]:
                # Stack predictions and take the mean across the predictions for each feature
                output[:, :, i:i+1] = torch.mean(torch.stack(predictions[i], dim=0), dim=0)
        
        return output
    
    
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in test_loader:
            input_x, seq_y = seq_x.to(device), seq_y.to(device)
            outputs = model(input_x)
            outputs = outputs.to(device)
            loss = criterion(outputs, seq_y)
            total_loss += loss.item() * input_x.size(0)
    return total_loss / len(test_loader.dataset)


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for seq_x, seq_y, seq_x_mark, seq_y_mark in train_loader:
        input_x, seq_y = seq_x.to(device), seq_y.to(device)
        optimizer.zero_grad()
        outputs = model(input_x)
        outputs = outputs.to(device)
        loss = criterion(outputs, seq_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_x.size(0)
    return total_loss / len(train_loader.dataset)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in val_loader:
            input_x, seq_y = seq_x.to(device), seq_y.to(device)
            outputs = model(input_x)
            outputs = outputs.to(device)
            loss = criterion(outputs, seq_y)
            total_loss += loss.item() * input_x.size(0)
    return total_loss / len(val_loader.dataset)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss


    
# seq_len, pred_len, n_points, dropout, wavelet_j, wavelet, subgraph_size, node_dim, n_gnn_layer
# Assuming DWT_MLP_Model is defined elsewhere, along with the necessary imports
seq_ = 24*4
pred_ = 24*4
n_points = 321
wavelet_j = 2
wavelet = 'haar'
dropout_rate = 0.05
supbraph_size = 6
node_dim = 40
n_gnn_layer = 3
s_size = 5
indices = [0,1,2,3,4,5] #np.random.choice(range(100), size=3, replace=False)
input_length = [24*4,512]



for i in indices:
    if i < 3:
        seq_length = input_length[0]
    else:
        seq_length = input_length[1]
        
    # Specify the file path
    root_path = '/home/choi/Wave_Transformer/optuna_/electricity/'
    data_path = 'electricity.csv'
    # Size parameters

    total_x = load_dataset(root_path, data_path, drop_columns=['date'])
    
    # Compute autocorrelation matrix
    autocorrelation_matrix = compute_autocorrelation(total_x)
    correlated_features = extract_correlated_features(autocorrelation_matrix)
    seq_len = seq_length # 24*4*4
    pred_len = 24*4
    #batch_size = bs
    # Initialize the custom dataset for training, validation, and testing
    train_dataset = Dataset_Custom(root_path=root_path, features= 'M', flag='train', data_path=data_path, step_size =s_size, size = [seq_len, pred_len])
    val_dataset = Dataset_Custom(root_path=root_path, features= 'M',flag='val', data_path=data_path,step_size = s_size)
    test_dataset = Dataset_Custom(root_path=root_path, features= 'M',flag='test', data_path=data_path,step_size = s_size)

    # Optionally, initialize the dataset for prediction (if needed)
    #pred_dataset = Dataset_Pred(root_path=root_path, flag='pred', size=size, data_path=data_path, inverse=True)

    # Example on how to create DataLoaders for PyTorch training (adjust batch_size as needed)
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last = False)
    dropout_rate = dropout_rate
    
    model = Model(
    seq_len=seq_len, 
    pred_len=pred_len, 
    n_points=n_points, 
    dropout=dropout_rate, 
    wavelet_j=wavelet_j, 
    wavelet=wavelet, 
    subgraph_size=supbraph_size, 
    node_dim=node_dim, 
    n_gnn_layer=n_gnn_layer,
    correlated_groups=correlated_features
)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    criterion = nn.MSELoss()
    
    # Assuming you have a device (GPU/CPU) setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    num_epochs = 50  # or any other number of epochs
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Call early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    test_loss = test(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')