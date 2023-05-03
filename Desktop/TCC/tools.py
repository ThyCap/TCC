import numpy as np
import torch
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

from pyDOE import lhs

# Setting Up Variables
T_avg = 0.5
T_low = 0
T_high = 1
T_mid = .3

R = 0.25
x_circle = 0.5
y_circle = 0.5
N_circle = 100

x_min, x_max, N_x = 0, 1, 100
y_min, y_max, N_y = 0, 1, 100

N_u = 200
N_f = 100_000

steps=1_000_000
lr=1e-1
layers = np.array([2,32,32,32,32,32,32,32,32,1]) #8 hidden layers

# Generate Data

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

# Boundary Conditions 
# define boundary conditions zones:
left_X = np.hstack((X[0,:][:,None], Y[0, :][:,None]))
left_T = T[0,:][:,None]*T_low

right_X = np.hstack((X[-1,:][:, None], Y[0,:][:, None]))
right_T = T[-1,:][:,None]*T_high

#define center circle
#perimeter points

circle_x = np.array([x_circle + R*np.cos(theta) for theta in np.linspace(0, 2*np.pi, N_circle)])
circle_y = np.array([y_circle + R*np.sin(theta) for theta in np.linspace(0, 2*np.pi, N_circle)])

circle_X = torch.from_numpy(np.transpose(np.vstack((circle_x, circle_y))))

circle_T = torch.ones((circle_X.shape[0], 1))*T_mid

X_test = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

#Domain bounds
lb = X_test[0] # lower bound
ub = X_test[-1] # upper bound 

X_train = np.vstack([left_X, right_X, circle_X])
T_train = np.vstack([left_T, right_T, circle_T])

# randomly choose N_u indices for training
idx = np.random.choice(X_train.shape[0], N_u + N_circle, replace = False)

X_train_Nu = X_train[idx, :]
T_train_Nu = T_train[idx, :]

# Latin Hypercube sampling for collocation points 
# N_f sets of tuples(x,t)
X_train_Nf = lb + (ub-lb)*lhs(2,N_f) 
X_train_Nf = np.vstack((X_train_Nf, X_train_Nu)) # append training points to collocation points 

