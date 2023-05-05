import numpy as np
import torch
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

from pyDOE import lhs
import time

#Set default dtype to float32
torch.set_default_dtype(torch.float64)

#PyTorch random number generator
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)

# Setting Up Variables
T_low, T_mid, T_high = 0, 0.3, 1

R, x_circle, y_circle, N_circle = 0.1, 0.5, 0.5, 500
x_min, x_max, N_x = 0, 1, 500
y_min, y_max, N_y = 0, 1, 500

N_u = 800
N_f = 100_000

steps=1_000
lr=1e-1
layers = np.array([2,48,48,48,48,48,48,48,48,1]) #8 hidden layers
tolerance = 1e-6

#Domain bounds
lb = float(x_min) # lower bound
ub = float(x_max) # upper bound 

# Generate Data
def generate_domain():
    x = torch.linspace(x_min, x_max, N_x).view(-1,1)
    y = torch.linspace(y_min, y_max, N_y).view(-1,1)

    x = x.float()
    y = y.float()

    X, Y = torch.meshgrid(x.squeeze(1), y.squeeze(1))

    T = torch.rand((len(x), len(y)),dtype = float)

    for i in range(len(x)):
        for j in range(len(y)):
            if x[i] == x_min:
                T[i][j] = T_low
            elif x[i] == x_max:
                T[i][j] = T_high

    T = T.float()

    return X, Y, T

# Generate circle perimeter points 
def generate_circle():
    circle_x = np.array([x_circle + R*np.cos(theta) for theta in np.linspace(0, 2*np.pi, N_circle)])
    circle_y = np.array([y_circle + R*np.sin(theta) for theta in np.linspace(0, 2*np.pi, N_circle)])

    circle_X = torch.from_numpy(np.transpose(np.vstack((circle_x, circle_y))))

    circle_T = torch.ones((circle_X.shape[0], 1))*T_mid

    return circle_X, circle_T

# Returns mask for elements of tensor outside a circle
def isNotInCircleTensorOrder1(Tensor):
    boolList = []

    for elem in Tensor:
        x, y = elem
        boolList.append((x - x_circle)**2 + (y - y_circle)**2 - R**2 > tolerance)

    return boolList

def isNotInCircleTensorOrder2(Tensor, x, y):
    boolList = []

    # If tensor order == 2, Tensor stores data in [i,j] position corresponding to x[i] and y[j]
    for i in range(Tensor.shape[0]):
        boolList_i = []
        for j in range(Tensor.shape[1]):
            boolList_i.append((x[i] - x_circle)**2 + (y[j] - y_circle)**2 - R**2 > tolerance)
        
        boolList.append(boolList_i)

    return np.array(boolList)

def generate_BC(X, Y, T, squareHasHole):
    # Boundary Conditions 
    # define boundary conditions zones:
    left_X = np.hstack((X[0,:][:,None], Y[0, :][:,None]))
    left_T = T[0,:][:,None]*T_low

    right_X = np.hstack((X[-1,:][:, None], Y[0,:][:, None]))
    right_T = T[-1,:][:,None]*T_high

    X_test = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    X_train = np.vstack((left_X, right_X))
    T_train = np.vstack((left_T, right_T))

    if squareHasHole:
        circle_X, circle_T = generate_circle()

        X_train = np.vstack((X_train, circle_X))
        T_train = np.vstack((T_train, circle_T))

    # randomly choose N_u indices for training
    idx = np.random.choice(X_train.shape[0], N_u + N_circle*squareHasHole, replace = False)

    X_train_Nu = X_train[idx, :]
    T_train_Nu = T_train[idx, :]

    return X_train, T_train, X_test, X_train_Nu, T_train_Nu

def generate_PDE(squareHasHole):
    # Latin Hypercube sampling for collocation points 
    # N_f sets of tuples(x,t)
    X_train_PDE = lb + (ub-lb)*lhs(2,N_f) 

    if squareHasHole:
        X_train_PDE = X_train_PDE[isNotInCircleTensorOrder1(X_train_PDE)]

        while X_train_PDE.shape[0] != N_f:
            current_N = X_train_PDE.shape[0]

            new_X_train_PDE = lb + (ub - lb)*lhs(2, N_f - current_N)
            X_train_PDE = np.vstack((X_train_PDE, new_X_train_PDE))

            X_train_PDE = X_train_PDE[isNotInCircleTensorOrder1(X_train_PDE)]

    return X_train_PDE

# Internal Heat Tensor
internalHeatTensor = torch.zeros((N_x, N_y))

# Run NN calculations
def NNCalculations(PINN):
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

    return u_pred