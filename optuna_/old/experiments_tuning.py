
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


from Model import *
from data_load import *


import optuna
import torch.optim as optim


import optuna
import torch.optim as optim

# %%
sequence_length = pred_length = 96

# %%

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
        #cols.remove('OT')
        #cols.remove('date')
        df_raw = df_raw[cols]
        # print(cols)
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


# %%
for X_batch, y_batch in train_loader:
    print(X_batch.shape, y_batch.shape)  # Process your batches here
    break

for X_batch, y_batch in valid_loader:
    print(X_batch.shape, y_batch.shape)  # Process your batches here
    break

for X_batch, y_batch in test_loader:
    print(X_batch.shape, y_batch.shape)  # Process your batches here
    break

# %%

# EarlyStopping class to monitor the validation loss
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Assuming `num_features` and `sentence_length` are defined based on your dataset
num_features = 321  # Example feature size (e.g., embedding dimension)
sentence_length = 96 # Example sentence length or sequence length
hidden_dim_factor = 2
model = UTransformerWithDWT(num_features=num_features, sentence_length=sentence_length, hidden_dim_factor= hidden_dim_factor, decompose_layer= 1, dropout=0.2922456519582798, num_layers=1, fixed=False)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00027233458898410953, weight_decay= 5.8363224331467285e-05)  # Added learning rate for clarity

# Lists for storing loss values
train_losses = []
valid_losses = []

epochs = 300

# Instantiate the early stopping
early_stopping = EarlyStopping(patience=50, verbose=True)

# Training loop with early stopping and conditional printing
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        predictions = model(data)
        loss = criterion(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for data, targets in valid_loader:
            predictions = model(data)
            loss = criterion(predictions, targets)
            valid_loss += loss.item()
    valid_loss /= len(valid_loader)
    valid_losses.append(valid_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')

    # Call early stopping
    early_stopping(valid_loss)
    if early_stopping.early_stop:
        print("Early stopping after epoch", epoch+1)
        break

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
test_loss = 0
for data, targets in test_loader:
    predictions = model(data)
    loss = criterion(predictions, targets)
    test_loss += loss.item()
test_loss /= len(test_loader)
print(f'Test Loss of DWT Skip Connection: {test_loss:.4f}')

print('The Best Test Loss So Far: ', 0.2583)

# %%
# The Best Params by Optuna
#Params: 
#    lr: 0.00027233458898410953
#    weight_decay: 5.8363224331467285e-05
#    hidden_dim_factor: 2
#    dropout_rate: 0.2922456519582798
#    decompose_layer: 1
#    batch_size: 64


# %%



