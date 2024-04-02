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
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
import pandas as pd

from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch.utils.data import DataLoader, TensorDataset, random_split


import optuna
import torch.optim as optim


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



class UTransformerWithDWT(nn.Module):
    def __init__(self, num_features, sentence_length,hidden_dim_factor, dropout, num_layers, wave='haar', mode='symmetric', decompose_layer=3, fixed=False):
        super(UTransformerWithDWT, self).__init__()
        self.dwt_forward = DWT1DForward(wave=wave, J=decompose_layer, mode=mode) 
        self.dwt_inverse = DWT1DInverse(wave=wave, mode=mode)

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

        seq_len = 96  # Adjust based on your specific application
        pred_out = pred_out[:, -seq_len:, :]  # Slicing to match the prediction sequence length

        return pred_out

