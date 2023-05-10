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

# Define type of problem
# 1. Simple diffusion in square
# 2. Diffusion in square with internal heat
# 3. Diffusion in square with circular hole
# 4. Combination ?
hasInternalHeat = False
squareHasHole = True

def partial_diff_equation(f, g):
    f_x_y = autograd.grad(f,g,torch.ones([g.shape[0], 1]), retain_graph=True, create_graph=True)[0] #first derivative
    f_xx_yy = autograd.grad(f_x_y,g,torch.ones(g.shape), create_graph=True)[0]#second derivative

    f_yy = f_xx_yy[:,[1]] # we select the 2nd element for y (the first one is x) (Remember the input X=[x,y]) 
    f_xx = f_xx_yy[:,[0]] # we select the 1st element for x (the second one is y) (Remember the input X=[x,y])

    u = f_xx + f_yy # loss equation
    u = u.float()

    return u

myProblem = Problem(partial_diff_equation, squareHasHole, hasInternalHeat)
myProblem.setTemp(T_left = 0, T_top = 0, T_right= 0, T_bottom= 0, T_circle= 1)

X_train_PDE, X_train_Nu, T_train_Nu, X_test = myProblem.getDomains()

PINN = FCN(myProblem, X_train_PDE, X_train_Nu, T_train_Nu, X_test, partial_diff_equation)

u_pred = myProblem.NNCalculations(PINN)

if not squareHasHole and not hasInternalHeat:
    torch.save(PINN.state_dict(), './PINN_simple.pt')
elif squareHasHole and not hasInternalHeat:
    torch.save(PINN.state_dict(), './PINN_holed.pt')
elif not squareHasHole and hasInternalHeat:
    torch.save(PINN.state_dict(), './PINN_heated.pt')
else:
    torch.save(PINN.state_dict(), './PINN_mixed.pt')
