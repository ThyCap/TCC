import torch
import torch.nn as nn

class sparseLayer(nn.Module):
  def __init__(self,PINN, n_in, n_out):
    super().__init__()

    self.n_in, self.n_out = n_in, n_out

    N = max(n_in, n_out)

    self.weights = [nn.Parameter(torch.zeros((1,1),dtype=torch.float32)) for i in range(N)]
    self.bias = [nn.Parameter(torch.zeros(1, dtype=torch.float32)) for i in range(N)]

    lim = 0.01
    for i in range(N):
        nn.init.uniform_(self.weights[i], -lim, lim)
        nn.init.zeros_(self.bias[i])
    
  def forward(self, x):
    xi = [x[:,i].reshape(-1, 1) for i in range(self.n_in)]

    if self.n_in > self.n_out:
        ratio = int(self.n_in//self.n_out)

        wx = [torch.mm(xi[i], self.weights[i].t()) for i in range(self.n_in)]
        ai = []

        for j in range(self.n_out):
                elem = torch.zeros((x.shape[0], 1), dtype= torch.float32)

                for k in range(ratio*j, ratio*(j + 1)):
                    elem = torch.add(elem, wx[k])
                    elem = torch.add(elem, self.bias[k])

                ai.append(elem)
    else:
        ratio = int(self.n_out//self.n_in)
        wx = [torch.mm(xi[i // ratio], self.weights[i].t()) for i in range(self.n_out)]
        ai = [torch.add(wx[i], self.bias[i]) for i in range(self.n_out)]

    if len(ai) == 1:
        result = torch.cat(ai, 1).reshape(-1,self.n_out)
    else:
        result = torch.cat(ai, 1).reshape(x.shape[0],self.n_out)

    return result