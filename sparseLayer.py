import torch
import torch.nn as nn

class sparseLayer(torch.nn.Module):
  def __init__(self, n_in, n_out):
    super().__init__()
    if n_in != 4 or n_out != 8:
      print("FATAL: Input skip layer must be (2,4) ")
    self.n_in, self.n_out = n_in, n_out  # (2,4)

    self.weights0 = torch.nn.Parameter(torch.zeros((2,1),dtype=torch.float32))
    self.weights1 = torch.nn.Parameter(torch.zeros((2,1),dtype=torch.float32))

    self.bias0 = torch.nn.Parameter(torch.tensor(2, dtype=torch.float32))
    self.bias1 = torch.nn.Parameter(torch.tensor(2, dtype=torch.float32))

    lim = 0.01
    torch.nn.init.uniform_(self.weights0, -lim, lim)
    torch.nn.init.uniform_(self.weights1, -lim, lim)

    torch.nn.init.zeros_(self.bias0)
    torch.nn.init.zeros_(self.bias1)

  def forward(self, x):
    # print("x="); print(x); print(x.shape);input()
    x0 = x[:,0].reshape(-1,1)
    # print("x0="); print(x0); print(x0.shape); input()
    x1 = x[:,1].reshape(-1,1)

    # print("self.weights0="); print(self.weights0);
    # print(self.weights0.shape); input()
    wx0= torch.mm(x0, self.weights0.t())
    wx1= torch.mm(x1, self.weights1.t())

    a = torch.add(wx0, self.bias0)
    b = torch.add(wx1, self.bias1)
   
    result = torch.cat((a, b), 1).reshape(-1,4)
    # print("result=");print(result);
    # print(result.shape); input()
    return result