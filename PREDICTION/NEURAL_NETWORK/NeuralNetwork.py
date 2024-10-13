import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import torch.nn.functional as F

# function for one hot encoding
def OneHotEncoding(T):
    OneHot_array = []
    for j in range(0, T.shape[1]):
        T_OneHot = F.one_hot(T[:,j].long())
        OneHot_array.append(T_OneHot)
    return torch.cat(OneHot_array, dim=1)

# read in data_____________________________________________________________________________________________________________________
input_data_path = '/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/NN_input_data_LEARN.txt'
data_raw = np.loadtxt(input_data_path, dtype = np.float32)  # the dtype must be float32, because of pytorch

# downscale the more prevalent class (class 0)
data_raw1 = data_raw[data_raw[:,-1] == 1.0,]
data_raw0 = data_raw[data_raw[:,-1] == 0.0,]
data0_small = data_raw0[0:305,:]
data = np.concatenate((data0_small, data_raw1), axis = 0)

#data = data_raw

X = data[:,:-1]
y = data[:,-1]
X = torch.tensor(X)
y = torch.tensor(y).reshape(-1, 1)

# one hot encode the matrix of predictors
X = OneHotEncoding(X)
X = X.type(torch.float32)

# Train-test split (of y and the one hot encoded X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Convert X_train and X_test  back to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

print('data ready: ')
print('X_train shape: {}'.format(X_train.shape))
print('X_test shape: {}'.format(X_test.shape))
print('y_train shape: {}'.format(y_train.shape))
print('y_test shape: {}'.format(y_test.shape))

input_dim = X_train.shape[1]   # could aswell be X_test.shape[1]

# define model architecture___________________________________________________________________________________________________

# play around with the model architecture

model = nn.Sequential(
        nn.Dropout(0.9),
        nn.Linear(input_dim, int(input_dim*0.8)),
        nn.ReLU(),
        nn.Linear(int(input_dim*0.8), int(input_dim*0.7)),
        nn.ReLU(),
        nn.Linear(int(input_dim*0.7),int(input_dim*0.6)),
        nn.ReLU(),
        nn.Linear(int(input_dim*0.6),int(input_dim*0.5)),
        nn.ReLU(),
        nn.Linear(int(input_dim*0.5),int(input_dim*0.4)),
        nn.ReLU(),
        nn.Linear(int(input_dim*0.4),int(input_dim*0.3)),
        nn.ReLU(),
        nn.Linear(int(input_dim*0.3),int(input_dim*0.2)),
        nn.ReLU(),
        nn.Linear(int(input_dim*0.2),int(input_dim*0.1)),
        nn.ReLU(),
        nn.Linear(int(input_dim*0.1),1),
        nn.Sigmoid()
        )


# define parameters___________________________________________________________________________________________________________________
# loss function and optimizer
#loss_fn = nn.BCELoss()  # binary cross entropy
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
n_epochs = 500    # number of epochs to run, # run for 30 epochs and then save it

# arrays for plotting
training_accuracy = np.zeros(len(range(0, n_epochs)), dtype = np.float64)
testing_accuracy = np.zeros(len(range(0, n_epochs)), dtype = np.float64)

print('learning...')

i = 0
for epoch in range(0, n_epochs):

    # forward pass
    y_pred_train = model(X_train)
    loss = loss_fn(y_pred_train, y_train)

    # backward pass
    optimizer.zero_grad()
    loss.backward()

    #update weigths
    optimizer.step()

    # training accuracy
    acc_train = (y_pred_train.round() == y_train).float().mean()

    # testing prediction
    y_pred_test = model(X_test)

    # testing accuracy
    acc_test = (y_pred_test.round() == y_test).float().mean()
    print("Epoch: {}".format(epoch))
    print("Training accuracy: {}".format(acc_train))
    print("Testing accuracy: {}".format(acc_test))
    training_accuracy[i] = acc_train
    testing_accuracy[i] = acc_test
    i = i + 1

print('learning done.')

print('plotting...')
plt.figure(figsize = (10,5))
plt.plot(range(0, n_epochs), training_accuracy, label='training_accuracy', color = 'red')
plt.plot(range(0, n_epochs), testing_accuracy, label='testing_accuracy', color = 'blue')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig('learning_curve.pdf')

print('saving model...')
torch.save(model, 'model.pt')
print('all done.')
