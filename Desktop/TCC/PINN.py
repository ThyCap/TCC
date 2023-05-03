from FCN import FCN
from tools import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

import time

#Set default dtype to float32
torch.set_default_dtype(torch.float64)

#PyTorch random number generator
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)

# 'Convert to tensor and send to GPU'
# X_train_Nf = torch.from_numpy(X_train_Nf).float()
# X_train_Nu = torch.from_numpy(X_train_Nu).float()
# U_train_Nu = torch.from_numpy(T_train_Nu).float()
# X_test = torch.from_numpy(X_test).float()
# u = torch.from_numpy(u_true).float()  
# f_hat = torch.zeros(X_train_Nf.shape[0],1)

'Neural Network Summary'

PINN = FCN(layers, X_train_Nf, X_train_Nu, T_train)
print(PINN)

params = list(PINN.parameters())

start_time = time.time()

optimizer = PINN.optimizer

optimizer.step(PINN.closure)

elapsed = time.time() - start_time                
print('Training time: %.2f' % (elapsed))


''' Model Accuracy ''' 
error_vec, u_pred = PINN.test()

print('Test Error: %.5f'  % (error_vec))

sns.heatmap(u_pred)