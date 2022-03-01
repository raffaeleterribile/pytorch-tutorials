# NumPy
import numpy as np
x = np.zeros((2, 3))

# PyTorch
import torch
y = torch.zeros(2, 3)

z = x + y

def funz(ball):
  return True

print(y.type()) # <class 'torch.FloatTensor'>
print(z.type()) # <class 'torch.DoubleTensor'>

xx = z.numpy()
xx += 1.0
print(z)
# 1  1  1
# 1  1  1
# [torch.DoubleTensor of size 2x3]

torch.Tensor([3, 2]) * torch.Tensor([[0, 1], [4, 2]])
# 0  2
# 12  4
# [torch.FloatTensor of size 2x2]

print(x.shape)   # (3, 2)
print(y.size())  # torch.Size([3, 2])

x.mean(axis=0)
y.mean(dim=0)

from torch.autograd import Variable
v = Variable(torch.ones(1, 2), requires_grad=True)

print(v.data)     #  1  1 [torch.FloatTensor of size 1x2]
print(v.grad)     # None
print(v.grad_fn)  # None

v_fn = torch.sum(v ** 2)
print(v_fn.data)    # 2 [torch.FloatTensor of size 1]
print(v_fn.grad_fn) # <SumBackward0 object at 0x7fa959f21550>

torch.autograd.grad(v_fn, v) # Gradiente di v_fn rispetto a v
# (Variable containing:
# 2  2 [torch.FloatTensor of size 1x2],)

v1 = Variable(torch.Tensor([1, 2]), requires_grad=True)
v2 = Variable(torch.Tensor([3]), requires_grad=True)
v_fn = torch.sum(v1 * v2)

v_fn.backward()
print(v1.grad) # Variable containing: 3 3 [torch.FloatTensor of size 2x1]
print(v2.grad) # Variable containing: 3 [torch.FloatTensor of size 1]

#----------------

import numpy as np

import torch

X = np.random.rand(30, 1)*2.0
w = np.random.rand(2, 1)
y = X*w[0] + w[1] + np.random.randn(30, 1) * 0.05

W = Variable(torch.rand(1, 1), requires_grad=True)
b = Variable(torch.rand(1), requires_grad=True)

def linear(x):
  return torch.matmul(x, W) + b

Xt = Variable(torch.from_numpy(X)).float()
yt = Variable(torch.from_numpy(y)).float()

for epoch in range(2500):

  # Calcola le predizioni
  y_pred = linear(Xt)

  # Calcola funzione costo
  loss = torch.mean((y_pred - yt) ** 2)

  # Esegui back-propagation
  loss.backward()

  # Aggiorna le variabili
  W.data = W.data - 0.005*W.grad.data
  b.data = b.data - 0.005*b.grad.data

  # Resetta il gradiente
  W.grad.data.zero_()
  b.grad.data.zero_()