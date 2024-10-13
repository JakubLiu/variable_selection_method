#!/usr/bin/python

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

def Missclassification(y_hat, y):
    res = (y_hat.round() == y).mean()
    return res


# read in and prepare data___________________________________________________________________________________________________
# load tensors
X = np.loadtxt('/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/DIM_REDUCTION/PCA/X_PC100_train.txt')
X = torch.tensor(X, dtype = torch.float32)
y = torch.load('/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/y_learn.pt')
y = torch.tensor(y, dtype = torch.float32)


# Train-test split (of y and the one hot encoded X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# define model architecture_______________________________________________________________________________________________

input_dim = X_train.shape[1]

class NN_Model(nn.Module):
    def __init__(self):
        super(NN_Model, self).__init__()
        self.dropout = nn.Dropout(0.2)
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




batch_size = 32
n_epochs = 60

dataset_train = TensorDataset(X_train, y_train)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataset_test = TensorDataset(X_test, y_test)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

model = NN_Model()
#loss_fn = losses.FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Get the dataset size for printing (it is equal to N_SAMPLES)
#dataset_size = len(dataloader.dataset)
#test_true = []
#test_predict = [] 

# Loop over epochs
for epoch in range(n_epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")

    # training loop
    model.train()
    train_score = []
    train_missclasification = []
    pred0 = 0
    pred1 = 0
    for id_batch, (x_batch, y_batch) in enumerate(dataloader_train):
        y_batch_pred = model(x_batch)
        loss = loss_fn(y_batch_pred, y_batch)
        train_score.append(f1_score(y_batch_pred.detach().numpy().round(), y_batch.detach().numpy()))
        train_missclasification.append(Missclassification(y_batch_pred.detach().numpy().round(), y_batch.detach().numpy()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        for y_pred in y_batch_pred.detach().numpy():
            if np.round(y_pred) == 0.0:
                pred0 = pred0 + 1
            else:
                pred1 = pred1 + 1
    

    # testing loop
    model.eval()
    test_score = []
    test_missclasification = []
    with torch.no_grad():

        for id_batch, (x_batch, y_batch) in enumerate(dataloader_test):
            y_batch_pred = model(x_batch)
            test_score.append(f1_score(y_batch_pred.detach().numpy().round(), y_batch.detach().numpy()))
            test_missclasification.append(Missclassification(y_batch_pred.detach().numpy().round(), y_batch.detach().numpy()))
            

    mean_train_score = np.round(np.mean(train_score),4)
    mean_test_score = np.round(np.mean(test_score),4)
    mean_train_missclassification = np.round(np.mean(train_missclasification),4)
    mean_test_missclasification = np.round(np.mean(test_missclasification),4)
    print('Train F1 score: {}       Test F1 score: {}    Train loss: {}\n'.format(mean_train_score, mean_test_score, loss))
    #print('Train missclasification: {}      Test missclasification: {}\n'.format(mean_train_missclassification, mean_test_missclasification))


print('class 0 count: {}    class 1 count: {}'.format(pred0, pred1))

'''
run = 4
print('saving model...')
torch.save(model, 'mode_PCA_100PC_{}epochs_run{}.pt'.format(n_epochs, run))
'''
print('all done.')