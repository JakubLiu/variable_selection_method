#!/usr/bin/python

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# read in and prepare data___________________________________________________________________________________________________
input_data_path = '/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/Encoded_input_data_learn.txt'
data = np.loadtxt(input_data_path, dtype = np.float32)  # the dtype must be float32, because of pytorch

X = data[:,:-1]
y = data[:,-1]
X = torch.tensor(X)
y = torch.tensor(y).reshape(-1, 1)

# Train-test split (of y and the one hot encoded X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_split = 21)


# define model architecture_______________________________________________________________________________________________

input_dim = X_train.shape[1]   # input_dim = 3979

class NN_Model(nn.Module):
    def __init__(self):
        super(NN_Model, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_dim, 228)
        self.fc2 = nn.Linear(228, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

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
        x = torch.relu(x)

        x = self.fc5(x)
        x = torch.sigmoid(x)

        return x


def Missclassification(y_hat, y):
    res = (y_hat.round() == y).float().mean()
    return res

def train_model(model, X_train, X_test, y_train, y_test, loss_function, optimizer, n_epochs):
    
    # evaluation on training dataset
    for epoch in range(0, n_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred_train = model(X_train)
        loss = loss_function(y_pred_train, y_train)
        loss.backward()
        optimizer.step()
        train_acc = Missclassification(y_pred_train, y_train)

        #evaluation on test dataset
        model.eval()
        test_loss = 0.0
        y_pred_test = model(X_test)
        loss = loss_function(y_pred_test, y_test)
        test_acc = Missclassification(y_pred_test, y_test)
        print("Epoch : {}/{}, trainig accuracy: {}, testing accuracy: {}".format(epoch+1, n_epochs, train_acc, test_acc))


model = NN_Model()
loss_fn = nn.BCELoss()  # binary cross entropy
optim = optim.Adam(model.parameters())
train_model(model, X_train, X_test, y_train, y_test, loss_function = loss_fn, optimizer = optim, n_epochs = 50)

'''
print('saving model...')
torch.save(model, 'model01.pt')
print('all done.')
'''

