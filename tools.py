import numpy as np
import torch

from pyDOE import lhs
import time

#Set default dtype to float32
torch.set_default_dtype(torch.float64)

#PyTorch random number generator
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)

class Problem:
    # Initialization function
    def __init__(self, partial_diff_equation, squareHasHole, weightsType = 'sqSized'):
        self.squareHasHole = squareHasHole
        self.weightsType = weightsType

        'Standard values variables'
        # Partial differential equation
        self.partial_diff_equation = partial_diff_equation

        # Temperature values
        self.setTemp()

        # BC type
        # Choose between 'Dirichlet', 'Neumann', None
        self.setBCtypes()

        # Domain definition variables
        ## Sides
        self.setDomainVars()
        
        ## Circle
        self.setCircleVars()

        ## Number of points in samples
        self.setSamplingVars()

        ## Optimizer and NN-related variables
        self.setNNVars()

        ## Domain bounds
        self.lb = float(self.x_min) # lower bound
        self.ub = float(self.x_max) # upper bound 

        # Define x and y
        self.x = torch.linspace(self.x_min, self.x_max, self.N_x, dtype= float).view(-1,1)
        self.y = torch.linspace(self.y_min, self.y_max, self.N_y, dtype= float).view(-1,1)
        
        self.dirichletMask = [] # list stores true values for Dirichlet BC points

    # temperature variables
    def setTemp(self, T_left = 0, T_top = 0, T_right = 1, T_bottom = 1, T_circle = 0.5):
        self.T_left = T_left
        self.T_top = T_top
        self.T_right = T_right
        self.T_bottom = T_bottom
        self.T_circle = T_circle

    # BC type
    def setBCtypes(self, BC_left = 'Dirichlet', BC_top = 'Dirichlet', BC_right = 'Dirichlet', BC_bottom = 'Dirichlet', BC_circle = 'Dirichlet'):
        self.BC_left = BC_left
        self.BC_top = BC_top
        self.BC_right = BC_right
        self.BC_bottom = BC_bottom
        self.BC_circle = BC_circle

    # domain variables
    def setDomainVars(self, x_min = 0,x_max = 1,N_x = 500,y_min = 0,y_max = 1,N_y = 500):
        self.x_min = x_min
        self.x_max = x_max
        self.N_x = N_x
        self.y_min = y_min
        self.y_max = y_max
        self.N_y = N_y

    # circle variables
    def setCircleVars(self, R = 0.1, x_circle = 0.5, y_circle = 0.5, N_circle = 500):
        self.R = R
        self.x_circle = x_circle
        self.y_circle = y_circle
        self.N_circle = N_circle

    # sampling variables
    def setSamplingVars(self, N_u = 1_000, N_f = 10_000):
        self.N_u = N_u
        self.N_f = N_f
    
    # neural network variables
    def setNNVars(self, steps = 1_000, lr = 1e-2, N_Layers = 2, Nodes = 32, tolerance = 1e-6):
        self.steps = steps
        self.lr = lr
        self.N_Layers = N_Layers
        self.Nodes = Nodes
        self.layers = np.hstack([[2], np.full(self.N_Layers,32), [1]]) #8 hidden layers
        self.tolerance = tolerance

    # generate domain function
    def generate_domain(self):
        x = self.x
        y = self.y

        x_min, x_max = x[0], x[-1]
        y_min, y_max = y[0], y[-1]

        X, Y = torch.meshgrid(x.squeeze(1), y.squeeze(1))

        T = torch.rand((len(x), len(y)),dtype = float)

        for i in range(len(x)):
            for j in range(len(y)):
                if x[i] == x_min:
                    T[i][j] = self.T_left
                elif x[i] == x_max:
                    T[i][j] = self.T_right
                elif y[j] == y_min:
                    T[i][j] = self.T_bottom
                elif y[i] == y_max:
                    T[i][j] = self.T_top

        T = T.float()

        self.X = X 
        self.Y = Y 
        self.T = T 

    # Generate circle perimeter points 
    def generate_circle(self):
        circle_x = np.array([self.x_circle + self.R*np.cos(theta) for theta in np.linspace(0, 2*np.pi, self.N_circle)])
        circle_y = np.array([self.y_circle + self.R*np.sin(theta) for theta in np.linspace(0, 2*np.pi, self.N_circle)])

        circle_X = torch.from_numpy(np.transpose(np.vstack((circle_x, circle_y))))

        circle_T = torch.ones((circle_X.shape[0], 1))*self.T_circle

        self.circle_X = circle_X
        self.circle_T = circle_T

    # Returns a order 1 mask for elements of tensor outside a circle
    def isNotInCircleTensorOrder1(self, Tensor):
        boolList = []

        for elem in Tensor:
            x, y = elem
            boolList.append((x - self.x_circle)**2 + (y - self.y_circle)**2 - self.R**2 > self.tolerance)

        return boolList

    # Returns a order 2 mask for elements of tensor outside a circle
    def isNotInCircleTensorOrder2(self, Tensor, x, y):
        boolList = []

        # If tensor order == 2, Tensor stores data in [i,j] position corresponding to x[i] and y[j]
        for i in range(Tensor.shape[0]):
            boolList_i = []
            for j in range(Tensor.shape[1]):
                boolList_i.append((x[i] - self.x_circle)**2 + (y[j] - self.y_circle)**2 - self.R**2 > self.tolerance)
            
            boolList.append(boolList_i)

        return np.array(boolList)

    # generate BC zones
    def generate_BC(self):
        X, Y = self.X, self.Y

        X_train_list = []
        T_train_list = []
        dirichletMask = []
        # Boundary Conditions 
        # define boundary conditions zones:
        if self.BC_left:
            left_X = np.hstack((X[0, :][:, None], Y[0, :][:, None]))
            left_T = np.full((left_X.shape[0], 1), self.T_left)

            X_train_list.append(left_X)
            T_train_list.append(left_T)
            dirichletMask = np.hstack((dirichletMask,(np.full(left_X.shape[0], self.BC_left == 'Dirichlet'))))

        if self.BC_top:
            top_X = np.hstack((X[1:-1, -1][:, None], Y[1:-1, -1][:, None]))
            top_T = np.full((top_X.shape[0], 1), self.T_top)

            X_train_list.append(top_X)
            T_train_list.append(top_T)
            dirichletMask = np.hstack((dirichletMask,(np.full(top_X.shape[0], self.BC_top == 'Dirichlet'))))

        if self.BC_right:
            right_X = np.hstack((X[-1, :][:, None], Y[0, :][:, None]))
            right_T = np.full((right_X.shape[0], 1), self.T_right)

            X_train_list.append(right_X)
            T_train_list.append(right_T)
            dirichletMask = np.hstack((dirichletMask,(np.full(right_X.shape[0], self.BC_right == 'Dirichlet'))))

        if self.BC_bottom:
            bottom_X = np.hstack((X[1:-1, 0][:, None], Y[1:-1, 0][:, None]))
            bottom_T = np.full((bottom_X.shape[0], 1), self.T_bottom)

            X_train_list.append(bottom_X)
            T_train_list.append(bottom_T)
            dirichletMask = np.hstack((dirichletMask,(np.full(bottom_X.shape[0], self.BC_bottom == 'Dirichlet'))))

        X_train = np.vstack(X_train_list)
        T_train = np.vstack(T_train_list)

        X_test = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

        if self.squareHasHole:
            self.generate_circle()

            X_train = np.vstack((X_train, self.circle_X))
            T_train = np.vstack((T_train, self.circle_T))
            dirichletMask = np.hstack((dirichletMask,(np.full(self.circle_X.shape[0], self.BC_circle == 'Dirichlet'))))

        # randomly choose N_u indices for training
        idx = np.random.choice(X_train.shape[0], self.N_u + self.N_circle*self.squareHasHole, replace = False)

        X_train_Nu = X_train[idx, :]
        T_train_Nu = T_train[idx, :]
        dirichletMask_Nu = np.array(dirichletMask[idx], dtype= bool)

        self.X_train, self.T_train = X_train, T_train
        self.X_test = X_test
        self.X_train_Nu, self.T_train_Nu = X_train_Nu, T_train_Nu
        self.dirichletMask = dirichletMask_Nu

    # generate PDE zones
    def generate_PDE(self):
        lb, ub, N_f = self.lb, self.ub, self.N_f
        # Latin Hypercube sampling for collocation points 
        # N_f sets of tuples(x,t)
        X_train_PDE = lb + (ub-lb)*lhs(2,N_f) 

        if self.squareHasHole:
            X_train_PDE = X_train_PDE[self.isNotInCircleTensorOrder1(X_train_PDE)]

            while X_train_PDE.shape[0] != N_f:
                current_N = X_train_PDE.shape[0]

                new_X_train_PDE = lb + (ub - lb)*lhs(2, N_f - current_N)
                X_train_PDE = np.vstack((X_train_PDE, new_X_train_PDE))

                X_train_PDE = X_train_PDE[self.isNotInCircleTensorOrder1(X_train_PDE)]

        self.X_train_PDE = X_train_PDE

    # return domain lists
    def getDomains(self):
        self.generate_domain()
        self.generate_BC()
        self.generate_PDE()

        return self.X_train_PDE, self.X_train_Nu, self.T_train_Nu, self.X_test

    # Run NN calculations
    def NNCalculations(self, PINN):
        print(PINN)

        params = list(PINN.parameters())

        start_time = time.time()
        optimizer = PINN.optimizer
        optimizer.step(PINN.closure)
        elapsed = time.time() - start_time                
        print('Training time: %.2f' % (elapsed))

        ''' Model Accuracy ''' 
        u_pred, lossHistoryTensor = PINN.test()

        return u_pred, lossHistoryTensor