from FCN import FCN
from tools import *

import torch
import torch.autograd as autograd         # computation graph

fname = './PINN_files/PINN_'
suffix = 'simple_horizontal'

# Define type of problem
# 1. Simple diffusion in square
# 2. Diffusion in square with internal heat
# 3. Diffusion in square with circular hole
# 4. Combination ?
squareHasHole = False
# internal heat to be implemented

## Possibilities for weight system
# 1. simple: weights = [1,1]
# 2. sized: weights = [N_u, N_f]
# 3. sqSized: weights = [sqrt(N_u), sqrt(N_f)]
# 4. sized_inv: weights = [N_u, N_f]
# 5. sqSized_inv: weights = [sqrt(N_u), sqrt(N_f)]
weightsType = 'sqSized_inv' 

def partial_diff_equation(f, g):
    f_x_y = autograd.grad(f,g,torch.ones([g.shape[0], 1]), retain_graph=True, create_graph=True)[0] #first derivative
    f_xx_yy = autograd.grad(f_x_y,g,torch.ones(g.shape), create_graph=True)[0]#second derivative

    f_yy = f_xx_yy[:,[1]] # we select the 2nd element for y (the first one is x) (Remember the input X=[x,y]) 
    f_xx = f_xx_yy[:,[0]] # we select the 1st element for x (the second one is y) (Remember the input X=[x,y])

    u = f_xx + f_yy # loss equation
    u = u.float()

    return u

myProblem = Problem(partial_diff_equation, squareHasHole, weightsType)
# myProblem.setTemp(T_left = 0, T_top = 0, T_right= 0, T_bottom= 0, T_circle= 1)
myProblem.setSamplingVars(N_u = 1000, N_f = 100_000)
myProblem.setNNVars(steps = 10_000, lr = 1e-3, tolerance = 1e-10, N_Layers = 1, Nodes = 8)
myProblem.setBCtypes(BC_left = 'Dirichlet', BC_top = 'Neumann', BC_right = 'Dirichlet', BC_bottom = 'Neumann')
myProblem.setTemp(T_left = 0, T_top = 0, T_right= 1, T_bottom= 0, T_circle= 1)

X_train_PDE, X_train_Nu, T_train_Nu, X_test = myProblem.getDomains()

PINN = FCN(myProblem, X_train_PDE, X_train_Nu, T_train_Nu, X_test, partial_diff_equation)

u_pred, [lossHistory, lossBCHistory, lossPDEHistory]= myProblem.NNCalculations(PINN)

torch.save(PINN.state_dict(), fname + suffix + '.pt')

np.savetxt('./history_files/loss_history_' + suffix + '.csv', np.asarray(lossHistory), delimiter=',')
np.savetxt('./history_files/loss_bc_history_' + suffix + '.csv', np.asarray(lossBCHistory), delimiter=',')
np.savetxt('./history_files/loss_pde_history_' + suffix + '.csv', np.asarray(lossPDEHistory), delimiter=',')

print("Saved to :" + fname + suffix + ".pt")