import numpy as np
import torch
import torch.nn as nn


def ConfusionMatrix(y, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(0, len(y)):
        if y[i] == 0.0 and y_pred[i] == 0.0:
            TN = TN + 1
        elif y[i] == 1.0 and y_pred[i] == 1.0:
            TP = TP + 1
        elif y[i] == 0.0 and y_pred[i] == 1.0:
            FP = FP + 1
        else:
            FN = FN + 1
    
    conf_mat = np.array([[TP,FP],
                         [FN,TN]])
    
    F1_score = TP/(TP + 0.5*(FP + FN))
    
    return conf_mat, F1_score

X = np.loadtxt('/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/DIM_REDUCTION/PCA/X_PC100_validate.txt')
X = torch.tensor(X, dtype = torch.float32)
y = torch.load('/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/y_validate.pt')
y = torch.tensor(y, dtype = torch.float32)

input_dim = X.shape[1]

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




# Load the model
model = NN_Model()
model = torch.load('/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/mode_PCA_100PC_60epochs_run4.pt')
model.eval()  # Set the model to evaluation mode
y_pred = model(X)
y_pred = y_pred.detach().numpy().round()
conf_mat, f1_score = ConfusionMatrix(y, y_pred)
print(conf_mat)
print(f1_score)
