from FCN import FCN
from tools import *

import torch
import torch.autograd as autograd         # computation graph

fname = './iter_evolution_study/PINN_'

# Define type of problem
# 1. Simple diffusion in square
# 2. Diffusion in square with internal heat
# 3. Diffusion in square with circular hole
# 4. Combination ?
squareHasHole = True
# internal heat to be implemented

## Possibilities for weight system
# 1. simple: weights = [1,1]
# 2. sized: weights = [N_u, N_f]
# 3. sqSized: weights = [sqrt(N_u), sqrt(N_f)]
# 4. evolutiveSized: weights = [sqrt(N_u)*(1 - exp(-5*t)), sqrt(N_f)*exp(-5*t)]
# 5. evolutiveSimple: weights = [(1 - exp(-5*t)), exp(-5*t)]
weightsType = 'sized' 

def partial_diff_equation(f, g):
    f_x_y = autograd.grad(f,g,torch.ones([g.shape[0], 1]), retain_graph=True, create_graph=True)[0] #first derivative
    f_xx_yy = autograd.grad(f_x_y,g,torch.ones(g.shape), create_graph=True)[0]#second derivative

    f_yy = f_xx_yy[:,[1]] # we select the 2nd element for y (the first one is x) (Remember the input X=[x,y]) 
    f_xx = f_xx_yy[:,[0]] # we select the 1st element for x (the second one is y) (Remember the input X=[x,y])

    u = f_xx + f_yy # loss equation
    u = u.float()

    return u

myProblem = Problem(partial_diff_equation, squareHasHole, weightsType)
myProblem.BCbooleans(BC_left = True, BC_top = True, BC_right = True, BC_bottom = True)
myProblem.setTemp(T_left = 0, T_top = 0, T_right= 0, T_bottom= 0, T_circle= 1)

suffix = 'wHole_'

X_train_PDE, X_train_Nu, T_train_Nu, X_test = myProblem.getDomains()

PINN = FCN(myProblem, X_train_PDE, X_train_Nu, T_train_Nu, X_test, partial_diff_equation)

u_pred, [lossHistory, lossBCHistory, lossPDEHistory] = myProblem.NNCalculations(PINN)

torch.save(PINN.state_dict(), fname + suffix + '.pt')

print("Saved to :" + fname + suffix + ".pt")
