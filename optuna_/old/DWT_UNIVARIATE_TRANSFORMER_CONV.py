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
from torch.nn import InstanceNorm1d
from torch import Tensor

import optuna


# %%
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
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
# Define the root path where your dataset is located and the name of your dataset file
root_path = '/home/choi/Wave_Transformer/optuna_/electricity/'
data_path = 'electricity.csv'

# Define the size configuration for your dataset
seq_len = 24 * 4    # Length of input sequences
label_len = 24 * 4      # Length of labels within the sequence to predict
pred_len = 24 * 4       # Number of steps to predict into the future

size = [seq_len, label_len, pred_len]

# Initialize the custom dataset for training, validation, and testing
train_dataset = Dataset_Custom(root_path=root_path, flag='train', size=size, data_path=data_path, step_size = 16)
val_dataset = Dataset_Custom(root_path=root_path, flag='val', size=size, data_path=data_path,step_size = 16)
test_dataset = Dataset_Custom(root_path=root_path, flag='test', size=size, data_path=data_path,step_size = 16)

# Optionally, initialize the dataset for prediction (if needed)
pred_dataset = Dataset_Pred(root_path=root_path, flag='pred', size=size, data_path=data_path, inverse=True)

# Example on how to create DataLoaders for PyTorch training (adjust batch_size as needed)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last = True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last = True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last = False)


# %%
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.norm = InstanceNorm1d(out_channels)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class DilatedTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(DilatedTCNBlock, self).__init__()
        
        # Calculate padding based on kernel size and dilation to maintain input length
        padding = (dilation * (kernel_size - 1)) // 2
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = InstanceNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
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
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
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
            x, cross, cross,
            attn_mask=cross_mask
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



# %%
class DWT_MLP_Model(nn.Module):
    def __init__(self, input_channels, seq_length, mlp_hidden_size, output_channels, decompose_layers=3, wave='haar', mode='symmetric', nhead=8, d_model=None, num_encoder_layers=3, dropout=0.1):
        super(DWT_MLP_Model, self).__init__()
        self.dwt_forward = DWT1DForward(J=decompose_layers, wave=wave, mode=mode)
        self.dwt_inverse = DWT1DInverse(wave=wave, mode=mode)
        self.seq_len = seq_length
        self.dropout = dropout
        if d_model is None:
            d_model = output_channels

        # Convolutional and TCN blocks for low and high-frequency components
        self.conv_low = ConvBlock(input_channels, output_channels)
        self.tcn_low = DilatedTCNBlock(input_channels, output_channels, dilation=2)
        
        self.conv_high_list = nn.ModuleList([ConvBlock(input_channels, output_channels) for _ in range(decompose_layers)])
        self.tcn_high_list = nn.ModuleList([DilatedTCNBlock(input_channels, output_channels, dilation=2) for _ in range(decompose_layers)])

        # Fully connected layers
        self.fc_low = nn.Linear(output_channels, mlp_hidden_size)
        self.fc_high_list = nn.ModuleList([nn.Linear(output_channels, mlp_hidden_size) for _ in range(decompose_layers)])
        
        # Positional Encoding
        self.pos_encoder = ProjectedPositionalEncoding(input_channels, d_model, max_len=5000)

        # Custom Transformer Encoder setup
        self.attention = attention  # Assuming attention is defined elsewhere
        encoder_layers_low = [EncoderLayer(self.attention, d_model, d_ff=mlp_hidden_size, dropout=self.dropout, activation="relu") for _ in range(num_encoder_layers)]
        self.transformer_low = Encoder(encoder_layers_low)

        # Transformer Encoders for high-frequency components, using custom Encoder
        self.transformer_high_list = nn.ModuleList(
            [Encoder([EncoderLayer(self.attention, d_model, d_ff=mlp_hidden_size, dropout=dropout, activation="relu") for _ in range(num_encoder_layers)]) 
             for _ in range(decompose_layers)])

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Adjust dimensions for DWT
        x_low, x_highs = self.dwt_forward(x)
       
        # Process low-frequency components
        #x_low_conv = self.conv_low(x_low)
        x_low_tcn = self.tcn_low(x_low)
        #x_low_combined = (x_low_conv + x_low_tcn) / 2
        x_low_combined = x_low_tcn
        #x_low_combined = self.fc_low(x_low_combined.permute(0, 2, 1))
        x_low_combined = x_low + x_low_combined
        x_low_combined = x_low_combined.permute(0,2,1)
        x_low_combined = self.pos_encoder(x_low_combined)
        x_low_combined, _ = self.transformer_low(x_low_combined) # Adjusted for custom encoder
        x_low_combined = x_low_combined.permute(0,2,1)
        x_low_combined = x_low + x_low_combined
        
        # Process high-frequency components
        x_highs_processed = []
        for i, x_high in enumerate(x_highs):
            #x_high_conv = self.conv_high_list[i](x_high)
            x_high_tcn = self.tcn_high_list[i](x_high)
            #x_high_combined = (x_high_conv + x_high_tcn) / 2
            x_high_combined = x_high_tcn
           
            x_high_combined = x_high + x_high_combined
            x_high_combined = x_high_combined.permute(0,2,1)
            #x_high_combined = self.fc_high_list[i](x_high_combined.permute(0, 2, 1))
            x_high_combined = self.pos_encoder(x_high_combined)
            x_high_combined, _ = self.transformer_high_list[i](x_high_combined)  # Adjusted for custom encoder
            x_high_combined = x_high.permute(0,2,1) + x_high_combined
            x_highs_processed.append(x_high_combined.permute(0,2,1))
        
        # Reconstruct the signal and adjust dimensions
        pred_out = self.dwt_inverse((x_low_combined, x_highs_processed)).permute(0, 2, 1)
        pred_out = pred_out[:, -self.seq_len:, :]
        
        return pred_out

# %%
# Initialize model, loss, and optimizer
model = DWT_MLP_Model(input_channels=321+4, seq_length = 96, mlp_hidden_size=64, output_channels=321+4)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for seq_x, seq_y, seq_x_mark, seq_y_mark in train_loader:
        inputs = torch.cat((seq_x, seq_x_mark), dim=-1)
        targets = torch.cat((seq_y, seq_y_mark), dim=-1)
       
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
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
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    
    # Print losses every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


# Save train and validation losses to a text file
with open('losses.txt', 'w') as f:
    f.write("Epoch,Train Loss,Validation Loss\n")
    for epoch in range(num_epochs):
        f.write(f"{epoch+1},{train_losses[epoch]},{val_losses[epoch]}\n")

# Plotting and saving the plot as PNG
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.close()


# %%
model.eval()
test_predictions = []
ground_truth = []
test_loss = 0
with torch.no_grad():
    for seq_x, seq_y, seq_x_mark, seq_y_mark in test_loader:
        inputs = torch.cat((seq_x, seq_x_mark), dim=-1)
        targets = torch.cat((seq_y, seq_y_mark), dim=-1).permute(0,2,1)
        outputs = model(inputs)
        test_loss += loss.item()
        test_predictions.extend(outputs.cpu().numpy())
        ground_truth.extend(targets.cpu().numpy())
test_loss /= len(test_loader)
print(f'Test Loss of DWT Skip Connection: {test_loss:.4f}')
# For simplicity, visualization is shown for one feature and one batch
if isinstance(ground_truth[0], torch.Tensor):
    ground_truth = [gt.numpy() for gt in ground_truth]
    test_predictions = [tp.numpy() for tp in test_predictions]

plt.figure(figsize=(15, 9))

# Plot the first three features
for feature_idx in range(3):
    plt.subplot(3, 1, feature_idx + 1)
    plt.plot(ground_truth[0][:, feature_idx], label=f'Ground Truth Feature {feature_idx + 1}')
    plt.plot(test_predictions[0][:, feature_idx], label=f'Predicted Feature {feature_idx + 1}')
    plt.legend()
    plt.title(f'Feature {feature_idx + 1}: Ground Truth vs Predicted')

plt.tight_layout()
plt.show()

# Save the test loss to a text file
with open('test_loss.txt', 'w') as f:
    f.write(f"Test Loss of DWT Skip Connection: {test_loss:.4f}")

# Saving plots for the first three features
for feature_idx in range(3):
    plt.figure(figsize=(15, 5))
    plt.plot(ground_truth[0][:, feature_idx], label=f'Ground Truth Feature {feature_idx + 1}')
    plt.plot(test_predictions[0][:, feature_idx], label=f'Predicted Feature {feature_idx + 1}')
    plt.legend()
    plt.title(f'Feature {feature_idx + 1}: Ground Truth vs Predicted')
    plt.savefig(f'feature_{feature_idx + 1}_comparison.png')
    plt.close()


