#!/usr/bin/python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import torch.nn.functional as F

'''
def OneHotEncoding(T):
    OneHot_array = []
    for j in range(0, T.shape[1]):
        T_OneHot = F.one_hot(T[:,j].long())
        OneHot_array.append(T_OneHot)
    return torch.cat(OneHot_array, dim=1)
'''

input_data_path = '/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/NN_input_data_FULL.txt'
data = np.loadtxt(input_data_path, dtype = np.int8)  # the dtype must be float32, because of pytorch
X = data[:,:-1]
y = data[:,-1]
X = torch.tensor(X)
y = torch.tensor(y).reshape(-1, 1)
#X = OneHotEncoding(X)
X = F.one_hot(X.to(torch.int64), num_classes = 3)
X = X.type(torch.float32)

'''
data_encoded = np.concatenate((X,y), axis = 1)
np.random.RandomState.seed(seed=21)
np.random.shuffle(data_encoded)
k = int(data_encoded.shape[0]*0.8)
data_encoded_learn = data_encoded[:k,:]
data_encoded_validate = data_encoded[k:,:]
np.savetxt('/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/Encoded_input_data_learn.txt', data_encoded_learn)
np.savetxt('/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/Encoded_input_data_validate_imbalanced.txt', data_encoded_validate)
data0 = data_encoded_validate[data_encoded_validate[:,-1] == 0.0,:]
data1 = data_encoded_validate[data_encoded_validate[:,-1] == 1.0,:]
data0_downsampled = data0[:data1.shape[0],:]
data_encoded_validate_balanced = np.concatenate((data0_downsampled, data1), axis = 0)
np.savetxt('/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/Encoded_input_data_validate_balanced.txt', data_encoded_validate_balanced)


print('Learn: {}'.format(data_encoded_learn.shape))
print('Validate imbalanced: {}'.format(data_encoded_validate.shape))
print('Validate balanced: {}'.format(data_encoded_validate_balanced.shape))
'''

torch.save(X, 'X_one_hot_tensor.pt')
torch.save(y, 'y_tensor.pt')
print('all done.')
