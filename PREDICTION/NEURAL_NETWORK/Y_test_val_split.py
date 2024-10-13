#!/usr/bin/python
import numpy as np
import torch
y = torch.load('/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/y_tensor.pt')
y_learn = y[:844] # for learning
y_validate = y[844:]  # for validation
torch.save(y_learn, '/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/y_learn.pt')
torch.save(y_validate, '/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/y_validate.pt')
print('all done.')