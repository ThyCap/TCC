import torch
import torch.autograd as autograd         # computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

from tools import *

class FCN(nn.Module):
    "Defines a connected network"
    
    def __init__(self, layers, x_domain, x_boundary, y_boundary, partial_diff_equation):
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

        # self.loss_function = nn.MSELOSS(reduction = 'mean')
        super().__init__() #call __init__ from parent class 

        self.x_domain = x_domain
        self.x_boundary = x_boundary
        self.y_boundary = y_boundary
              
        'activation function'
        self.activation = nn.Tanh()

        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')

        'loss value storage'
        self.loss_bc_history = []
        self.loss_pde_history = []
        self.error_vec_history = []
    
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        self.iter = 0
        self.float()

        self.partial_diff_equation = partial_diff_equation

        'L-BFGS Optimizer'
        self.optimizer = optim.LBFGS(self.parameters(), lr, 
                              max_iter = steps, 
                              max_eval = None, 
                              tolerance_grad = 1e-14, 
                              tolerance_change = 1e-14, 
                              history_size = steps, 
                              line_search_fn = 'strong_wolfe')
    
        'Xavier Normal Initialization'
        for i in range(len(layers)-1):
            
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)


    def forward(self, x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
        
        u_b = torch.from_numpy(ub).float()
        l_b = torch.from_numpy(lb).float()
                      
        #preprocessing input 
        x = (x - l_b)/(u_b - l_b) #feature scaling
        
        #convert to float
        a = x.float()
        
        for i in range(len(layers)-2):
            
            z = self.linears[i](a)
                        
            a = self.activation(z)
            
        a = self.linears[-1](a)
        
        return a.float()

    def loss_BC(self, x_BC, y_BC):
        y_BC = torch.from_numpy(y_BC).float()
        y_forward_BC = self.forward(x_BC).float()
        loss_bc = self.loss_function(y_forward_BC, y_BC)
        loss_bc = loss_bc.float()

        self.loss_bc_history.append(loss_bc.float())

        return loss_bc

    def loss_PDE(self, x_PDE):
        x_PDE = torch.from_numpy(x_PDE)

        g = x_PDE.clone()
        g.requires_grad = True

        f = self.forward(g)

        u = self.partial_diff_equation(f, g)

        u_hat = torch.zeros(x_PDE.shape[0],1)  
        u_hat = u_hat.float()

        loss = self.loss_function(u, u_hat)

        self.loss_bc_history.append(loss)

        return loss 
    
    def loss(self, x_BC, y_BC, x_PDE):
        loss_bc = self.loss_BC(x_BC, y_BC)
        loss_pde = self.loss_PDE(x_PDE)
        return N_f/N_u*loss_bc + loss_pde, loss_bc.item(), loss_pde.item()

    'callable for optimizer'                                       
    def closure(self):
        optimizer = self.optimizer
        optimizer.zero_grad()
        loss, loss_bc, loss_pde = self.loss(self.x_boundary, self.y_boundary, self.x_domain)
        loss.backward()

        self.iter += 1

        if self.iter % 50 == 1 or self.iter == 1:
            print("Iter \t\t Total Loss \t\t Loss per element \t Mean Loss_BC \t\t Mean Loss_PDE")

        if self.iter % 5 == 0:
            error_vec, _ = self.test()
            
            print("%i \t\t %.3e \t\t %.3e \t\t %.3e \t\t %.3e" % (self.iter, loss.item(),loss.item()/(N_x*N_y), loss_bc, loss_pde))

        return loss 

    def test(self):
        u_pred = self.forward(X_test)
        u = torch.zeros(u_pred.shape)
        error_vec = torch.linalg.norm((u-u_pred),2)      # L2 Norm of the error (Vector)
        u_pred = u_pred.cpu().detach().numpy()
        u_pred = np.reshape(u_pred,(100,100),order='F')
                
        return error_vec, u_pred 