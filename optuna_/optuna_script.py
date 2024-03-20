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
from Model2 import *
from data_load import *


import optuna
import torch.optim as optim


root_path = "/home/choi/Wave_Transformer/optuna_/electricity/"
data_path = 'electricity.txt'


# Instantiate datasets
train_dataset = Dataset_Custom(root_path, 'train', (seq_len, pred_len), data_path, scale=True)
val_dataset = Dataset_Custom(root_path, 'val', (seq_len, pred_len), data_path, scale=True)
test_dataset = Dataset_Custom(root_path, 'test', (seq_len, pred_len), data_path, scale=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last= True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle= False, drop_last = True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last = True)

def objective(trial):
    # Hyperparameters to be tuned
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    hidden_dim_factor = trial.suggest_int("hidden_dim_factor", 1, 8)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    decompose_layer = trial.suggest_int("decompose_layer", 1, 3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    #Shuffle_ = trial.suggest_categorical("shuffle", [True, False])
    embed_size = trial.suggest_categorical("embed_size", [64, 128, 256, 512, 1024])
    heads = trial.suggest_categorical('heads', [1, 2, 4, 6, 8])
	
    wave = 'haar'
    mode = 'symmetric'
    forward_expansion = hidden_dim_factor
    
    # Specify the file path
    root_path = "~/AD_Datasets/electricity/"
    data_path = 'electricity.txt'
    
    # Size parameters
    seq_len = 96
    pred_len = 96
    
    # Fixed parameters
    num_features = 321  # Adjust according to your dataset
    sentence_length = 96  # Adjust according to your dataset
    num_layers = decompose_layer
    
    try:
        # Model instantiation (ensure all necessary variables are defined)
        model = UTransformerWithDWT(num_features, embed_size, sentence_length, hidden_dim_factor, forward_expansion, dropout_rate, num_layers, wave, mode, decompose_layer)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.MSELoss()  # Choose according to your task

        # Example training loop (ensure train_loader is defined)
        model.train()
        epochs = 10
        for _ in range(epochs):
            for data, targets in train_loader:  # Ensure your data loading is correct
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()

        # Evaluation loop (ensure valid_loader is defined)
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for data, targets in valid_loader:
                output = model(data)
                loss = criterion(output, targets)
                valid_loss += loss.item()

        # Normalize the validation loss
        valid_loss /= len(valid_loader)

    except Exception as e:
        # Here, you can decide what to do in case of an exception.
        # For instance, you might want to return a high loss value
        # so that this trial is considered a bad one.
        print(f"An exception occurred: {e}")
        valid_loss = float('inf')  # Assigning a high loss value

    return valid_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)  # Example: Run 20 trials

print("Best trial:")
trial = study.best_trial

print(f"Value: {trial.value}")
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Save the study results to a text file
results_file_path = "optuna_study_results_transformer.txt"
with open(results_file_path, "w") as file:
    file.write("Best trial:\n")
    file.write(f"Value: {trial.value}\n")
    file.write("Params:\n")
    for key, value in trial.params.items():
        file.write(f"    {key}: {value}\n")
    file.write("\nFull study results:\n")
    for trial in study.trials:
        file.write(f"Trial {trial.number}: Value: {trial.value}, Params: {trial.params}\n")

print(f"Optuna study results saved to {results_file_path}")
