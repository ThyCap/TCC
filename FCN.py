import torch
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
import torch.autograd as autograd

import time

import numpy as np
from tools import *
from sparseLayer import *

device = torch.device('cpu')

class FCN(nn.Module):
    "Defines a connected network"
    def __init__(self, Problem, x_domain, x_boundary, y_boundary, x_test, partial_diff_equation):
        # super().__init__()
        # activation = nn.Tanh
        # self.fcs = nn.Sequential(*[
        #                 nn.Linear(N_INPUT, N_HIDDEN),
        #                 activation()])
        # self.fch = nn.Sequential(*[
        #                 nn.Sequential(*[
        #                     nn.Linear(N_HIDDEN, N_HIDDEN),
        #                     activation()]) for _ in range(N_LAYERS-1)])
        # self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

        super().__init__() #call __init__ from parent class 

        self.Problem = Problem
        self.x_domain = x_domain
        self.x_boundary = x_boundary
        self.y_boundary = y_boundary
        self.x_test = x_test
              
        'activation function'
        self.activation = nn.Tanh()

        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')

        'loss value storage'
        self.loss_bc_history = []
        self.loss_history = []
        self.loss_pde_history = []
        self.error_vec_history = []
    
        'Initialise neural network as a list using nn.Modulelist' 
        self.sparseLayer = sparseLayer(n_in = 2, n_out = 8)
        self.hiddenLayers = nn.ModuleList([nn.Linear(Problem.layers[i], Problem.layers[i+1]) for i in range(len(Problem.layers)-1)])
        
        self.iter = 0
        self.startTime = time.time()
        self.totalElapsedTimeHistory = [0]
        self.float()

        self.partial_diff_equation = partial_diff_equation

        'L-BFGS Optimizer'
        self.optimizer = optim.LBFGS(self.parameters(), 
                                    lr = Problem.lr, 
                                    max_iter = Problem.steps, 
                                    max_eval = 10_000, 
                                    tolerance_grad = 1e-14, 
                                    tolerance_change = 1e-14, 
                                    history_size = 10_000, 
                                    line_search_fn = 'strong_wolfe')
        # self.optimizer = optim.SGD(self.parameters(), 
        #                             lr = Problem.lr, 
        #                             momentum= 0.999)
    
        'Xavier Normal Initialization'
        for i in range(len(Problem.layers)-1):
            nn.init.xavier_normal_(self.hiddenLayers[i].weight.data, gain=1.0)
            
            # set biases to zero
            nn.init.zeros_(self.hiddenLayers[i].bias.data)

    def forward(self, x):
        lb, ub, layers = self.Problem.lb, self.Problem.ub, self.Problem.layers

        # print("result=");print(result);
        # print(result.shape); input()
        # return result

        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
                      
        #preprocessing input 
        x = (x - lb)/(ub - lb) #feature scaling
        
        #convert to float
        a = x.float()

        # pass through custom layer
        z = self.activation(self.sparseLayer(a))
        
        for i in range(len(layers)-2):
            z = self.hiddenLayers[i](z)         
            a = self.activation(z)
            
        a = self.hiddenLayers[-1](a)
        
        return a.float()

    def loss_BC_Dirichlet(self, x_BC, y_BC):
        y_BC = torch.from_numpy(y_BC).float()
        y_forward_BC = self.forward(x_BC).float()
        loss_bc = self.loss_function(y_forward_BC, y_BC)
        loss_bc = loss_bc.float()

        return loss_bc.item()
    
    def loss_BC_Neumann(self, x_BC, y_BC):
        y_BC = torch.from_numpy(y_BC).float()
        x_BC = torch.from_numpy(x_BC).float()

        g = x_BC.clone()
        g.requires_grad = True

        f = self.forward(g)

        f_x_y = autograd.grad(f,g,torch.ones([g.shape[0], 1]), retain_graph=True, create_graph=True, allow_unused = True)[0] #first derivative

        f_y = f_x_y[:,[1]] # we select the 2nd element for y (the first one is x) (Remember the input X=[x,y]) 
        f_x = f_x_y[:,[0]] # we select the 1st element for x (the second one is y) (Remember the input X=[x,y])

        u = 0*f_x + f_y # loss equation
        
        u = u.float()

        u_hat = torch.zeros(x_BC.shape[0],1, dtype = float)  
        loss = self.loss_function(u, u_hat)

        return loss.item()

    def loss_PDE(self, x_PDE):
        x_PDE = torch.from_numpy(x_PDE)

        g = x_PDE.clone()
        g.requires_grad = True

        f = self.forward(g)

        u = self.partial_diff_equation(f, g)

        u_hat = torch.zeros(x_PDE.shape[0],1)  
        u_hat = u_hat.float()

        loss = self.loss_function(u, u_hat)

        return loss 
    
    def loss(self, x_BC, y_BC, x_PDE):
        ## Possibilities for weight system
        # 1. simple: weights = [1,1]
        # 2. sized: weights = [N_u, N_f]
        # 3. sqSized: weights = [sqrt(N_u), sqrt(N_f)]
        # 4. evolutiveSized: weights = [sqrt(N_u)*(1 - exp(-5*t)), sqrt(N_f)*exp(-5*t)]
        # 5. evolutiveSimple: weights = [(1 - exp(-5*t)), exp(-5*t)]
        weights = [1, 1]

        if self.Problem.weightsType == 'sized':
            weights = [self.Problem.N_u, self.Problem.N_f]
        elif self.Problem.weightsType == 'sqSized':
            weights = [np.sqrt(self.Problem.N_u), np.sqrt(self.Problem.N_f)]
        elif self.Problem.weightsType == 'sized_inv':
            weights = [self.Problem.N_f, self.Problem.N_u]
        elif self.Problem.weightsType == 'sqSized_inv':
            weights = [np.sqrt(self.Problem.N_f), np.sqrt(self.Problem.N_u)]
        
        #normalize weights
        weights = np.array(weights)/sum(weights)

        x_BC_Dirichlet = x_BC[self.Problem.dirichletMask]
        y_BC_Dirichlet = y_BC[self.Problem.dirichletMask]
        x_BC_Neumann = x_BC[(1 - self.Problem.dirichletMask)]
        y_BC_Neumann = y_BC[(1 - self.Problem.dirichletMask)]

        loss_bc_dirichlet = self.loss_BC_Dirichlet(x_BC_Dirichlet, y_BC_Dirichlet)
        loss_bc_neumann = self.loss_BC_Neumann(x_BC_Neumann, y_BC_Neumann)
        loss_pde = self.loss_PDE(x_PDE)

        loss_bc = np.sqrt(loss_bc_dirichlet**2 + loss_bc_neumann**2) 

        loss = weights[0]*loss_bc + weights[1]*loss_pde

        self.loss_history.append(loss.item())
        self.loss_bc_history.append(loss_bc.item())
        self.loss_pde_history.append(loss_pde.item())

        return loss, loss_bc.item(), loss_pde.item()
    
    def lossTensor(self, x_Test):
        x_Test = torch.from_numpy(x_Test)

        g = x_Test.clone()
        g.requires_grad = True

        f = self.forward(g)

        u = self.partial_diff_equation(f, g)

        return u        

    'callable for optimizer'                                       
    def closure(self):
        N_x, N_y = self.Problem.N_x, self.Problem.N_y

        optimizer = self.optimizer
        optimizer.zero_grad()
        loss, loss_bc, loss_pde = self.loss(self.x_boundary, self.y_boundary, self.x_domain)
        loss.backward()

        self.iter += 1
        self.totalElapsedTimeHistory.append(time.time() - self.startTime)        

        if self.iter % 50 == 1 or self.iter == 1:
            print("Iter \t\t Combined Loss \t\t Mean Loss_BC \t\t Mean Loss_PDE \t\t Total Elapsed Time (s)")

        if self.iter % 1 == 0:

            # if self.iter % 100 == 0 and self.iter > 1:
            #     u_pred, lossHistoryTensor = self.test()
            #     lossTensor = self.lossTensor(self.x_test)

            #     lossTensor = lossTensor.detach().numpy()

            #     np.savetxt('./iter_evolution_study/u_pred/u_pred@' + str(self.iter) + '.csv', np.asarray(u_pred), delimiter=',')
            #     np.savetxt('./iter_evolution_study/lossTensor/lossTensor@' + str(self.iter) + '.csv', np.asarray(lossTensor), delimiter=',')

            print("%i \t\t %.3e \t\t %.3e \t\t %.3e \t\t %.3e" % (self.iter, loss.item(), loss_bc, loss_pde, self.totalElapsedTimeHistory[-1]))

        return loss 

    def test(self):
        N_x, N_y = self.Problem.N_x, self.Problem.N_y
        X_test = self.x_test

        u_pred = self.forward(X_test)
        u_pred = u_pred.cpu().detach().numpy()
        u_pred = np.reshape(u_pred,(N_x, N_y),order='F')

        lossHistoryTensor = [self.loss_history, self.loss_bc_history, self.loss_pde_history]
                
        return u_pred, lossHistoryTensor