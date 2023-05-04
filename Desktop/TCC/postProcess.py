import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

def full_heatmap(X, Y, PINN, u_pred):
    #Plot confusion matrix heatmap
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)

    physics_X = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    print(physics_X.shape)
    print(X.shape)
    print(Y.shape)
    physics_T = PINN.forward(physics_X).detach().numpy()


    sns.heatmap(u_pred,
                xticklabels = X,
                yticklabels = Y,
                cmap = 'coolwarm',
                annot = True,
                fmt = '.5g',
                vmax = 200)

    plt.xlabel('Predicted',fontsize=22)
    plt.ylabel('Actual',fontsize=22)
    plt.show()