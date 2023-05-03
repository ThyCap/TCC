import sys 

sys.path.insert(0, '/../TCC')

from FCN import FCN
from tools import *

import numpy as np
import seaborn as sns

from PIL import Image

import torch
import torch.autograd as autograd         # computation graph

import time

# 'Convert to tensor and send to GPU'
# X_train_Nf = torch.from_numpy(X_train_Nf).float()
# X_train_Nu = torch.from_numpy(X_train_Nu).float()
# U_train_Nu = torch.from_numpy(T_train_Nu).float()
# X_test = torch.from_numpy(X_test).float()
# u = torch.from_numpy(u_true).float()  
# f_hat = torch.zeros(X_train_Nf.shape[0],1)

'Neural Network Summary'

def partial_diff_equation(f, g):
    f_x_y = autograd.grad(f,g,torch.ones([g.shape[0], 1]), retain_graph=True, create_graph=True)[0] #first derivative
    f_xx_yy = autograd.grad(f_x_y,g,torch.ones(g.shape), create_graph=True)[0]#second derivative

    f_yy = f_xx_yy[:,[1]] # we select the 2nd element for y (the first one is x) (Remember the input X=[x,y]) 
    f_xx = f_xx_yy[:,[0]] # we select the 1st element for x (the second one is y) (Remember the input X=[x,y])

    u = f_xx + f_yy # loss equation
    u = u.float()

    return u

PINN = FCN(layers, X_train_Nf, X_train_Nu, T_train, partial_diff_equation)
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