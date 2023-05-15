from FCN import FCN
from tools import *

import torch
import torch.autograd as autograd         # computation graph

fname = './PINN_files/PINN_'


# Define type of problem
# 1. Simple diffusion in square
# 2. Diffusion in square with internal heat
# 3. Diffusion in square with circular hole
# 4. Combination ?
hasInternalHeat = False
squareHasHole = False

def partial_diff_equation(f, g):
    f_x_y = autograd.grad(f,g,torch.ones([g.shape[0], 1]), retain_graph=True, create_graph=True)[0] #first derivative
    f_xx_yy = autograd.grad(f_x_y,g,torch.ones(g.shape), create_graph=True)[0]#second derivative

    f_yy = f_xx_yy[:,[1]] # we select the 2nd element for y (the first one is x) (Remember the input X=[x,y]) 
    f_xx = f_xx_yy[:,[0]] # we select the 1st element for x (the second one is y) (Remember the input X=[x,y])

    u = f_xx + f_yy # loss equation
    u = u.float()

    return u

for N in [2,4,6,8]:
    myProblem = Problem(partial_diff_equation, squareHasHole, hasInternalHeat)
    # myProblem.setTemp(T_left = 0, T_top = 0, T_right= 0, T_bottom= 0, T_circle= 1)
    myProblem.setTemp(T_left = 0, T_top = 0.3, T_right= 1, T_bottom= 0.5, T_circle= 1)

    myProblem.setNNVars(N_Layers=4)
    suffix = 'test_' + str(N)

    X_train_PDE, X_train_Nu, T_train_Nu, X_test = myProblem.getDomains()

    PINN = FCN(myProblem, X_train_PDE, X_train_Nu, T_train_Nu, X_test, partial_diff_equation)

    u_pred, [lossHistory, lossBCHistory, lossPDEHistory] = myProblem.NNCalculations(PINN)

    torch.save(PINN.state_dict(), fname + suffix + '.pt')

    np.savetxt('./history_files/loss_history_' + suffix + '.csv', np.asarray(lossHistory), delimiter=',')
    np.savetxt('./history_files/loss_bc_history_' + suffix + '.csv', np.asarray(lossBCHistory), delimiter=',')
    np.savetxt('./history_files/loss_pde_history_' + suffix + '.csv', np.asarray(lossPDEHistory), delimiter=',')
