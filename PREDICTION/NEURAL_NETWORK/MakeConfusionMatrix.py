#!/usr/bin/python

import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

def BinaryConfusionMatrix(y, y_pred):
    mat = np.zeros((2,2), dtype = np.float32)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(0,y.shape[0]):
        if y[i] == 0 and y_pred[i] == 0:
            TN = TN + 1
        elif y[i] == 1 and y_pred[i] == 1:
            TP = TP + 1
        elif y[i] == 0 and y_pred[i] == 1:
            FP = FP + 1
        else:
            FN = FN + 1
        
    mat[0,0] = TN
    mat[1,1] = TP
    mat[1,0] = FN
    mat[0,1] = FP
    
    return mat


X = np.loadtxt('/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/DIM_REDUCTION/PCA/X_PC100.txt')
X = torch.tensor(X, dtype = torch.float32)
y = torch.load('/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/y_tensor.pt')
y = torch.tensor(y, dtype = torch.float32)

input_dim = X.shape[1]

class NN_Model(nn.Module):
    def __init__(self):
        super(NN_Model, self).__init__()
        #self.dropout = nn.Dropout(0.2)
        #self.LReLU = nn.LeakyReLU()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16,1)

    def forward(self, x):
        #x = x.view(x.shape[0], -1)  # Flatten the input
        
        #x = self.dropout(x)

        x = self.fc1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        
        x = self.fc3(x)
        x = torch.relu(x)

        x = self.fc4(x)
        x = torch.sigmoid(x)

        return x


model = NN_Model()
model = torch.load('/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/mode_PCA_100PC_60epochs.pt')
model.eval()  # Set the model to evaluation mode
y_pred = model(X)
y_pred = y_pred.detach().numpy()
y_pred = np.round(y_pred)
y = y.detach().numpy()
ConfusionMatrix = BinaryConfusionMatrix(y, y_pred)
print(ConfusionMatrix)