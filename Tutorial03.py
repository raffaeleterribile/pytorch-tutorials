import torch
from torch import nn
import torch.nn.functional as F

class Swish1(nn.Module):
    def forward(self, input):
        return input * F.sigmoid(input)

net_sequential = nn.Sequential(
        nn.Linear(4, 10),
        Swish1(),
        nn.Linear(10, 3)
)

# Valori sull'asse x
x = np.linspace(-5.0, 5.0, 1000).reshape(-1, 1)

# Calcola Swish-1 su tutti i valori
swish1 = Swish1()
y = swish1(torch.from_numpy(x))

# Grafica il risultato
plt.plot(x, y.numpy())

x = torch.linspace(-5.0, 5.0, 1000, requires_grad=True)
# Ci sono modi più efficienti! :-)
g = [torch.autograd.grad(swish1(xi), xi) for xi in x]

plt.plot(x.detach().numpy(), g)

# class ConstantBetaSwish(nn.Module):
#     # QUESTA IMPLEMENTAZIONE E' ERRATA

#     def __init__(self, beta=2.0):
#         super(ConstantBetaSwish, self).__init__()
#         self.beta = torch.tensor(beta)

#     def forward(self, input):
#         return input * F.sigmoid(input * self.beta)

# swish2 = ConstantBetaSwish()
# swish2.cuda()
# swish2(torch.tensor(3.0).cuda())

class ConstantBetaSwish(nn.Module):

    def __init__(self, beta=2.0):
        super(ConstantBetaSwish, self).__init__()
        self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))

    def forward(self, input):
        return input * F.sigmoid(input * Variable(self.beta))

    def extra_repr(self):
        return 'beta={}'.format(self.beta)

net = nn.Sequential(
        nn.Linear(4, 5),
        ConstantBetaSwish(),
        nn.Linear(5, 2)
)
print(net)
# Sequential(
#  (0): Linear(in_features=4, out_features=5, bias=True)
#  (1): ConstantBetaSwish(beta=2.0)
#  (2): Linear(in_features=5, out_features=2, bias=True)
# )

class BetaSwish(nn.Module):
    def __init__(self, num_parameters=1):
        super(BetaSwish, self).__init__()

        self.num_parameters = num_parameters
        self.beta = torch.nn.Parameter(torch.ones(1, num_parameters))

    def forward(self, input):
        return input * F.sigmoid(input * self.beta)

net = nn.Sequential(
        nn.Linear(4, 10),
        BetaSwish(10),
        nn.Linear(10, 3)
)

net[1].beta.detach().numpy()
# array([[0.44259977, 0.9548798 , 0.19685858, 1.0720267 , 3.1051192 ,
#        0.09515426, 1.9494272 , 1.4172938 , 1.4043328 , 0.06701402]],
#      dtype=float32)

from scipy.special import expit

class SwishFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        # Calcola la sigmoide (uscendo da autograd)
        input_sigmoid = torch.from_numpy(expit(input.detach().numpy()))
        # Salva tutto quello che serve per la back-propagation
        ctx.save_for_backward(input, input_sigmoid)
        return input * input_sigmoid

    @staticmethod
    def backward(ctx, grad_output):
        # Recupera i tensori salvati
        input,input_sigmoid, = ctx.saved_tensors
        # Calcola il gradiente
        grad_af = input_sigmoid + input * input_sigmoid * (1 - input_sigmoid)
        return grad_output * grad_af

from torch.autograd import gradcheck
input = (torch.randn(20, 20, requires_grad=True),)
test = gradcheck(SwishFunction.apply, input, eps=1e-2, atol=1e-2)
print(test)
# True

swish = SwishFunction.apply

class Swish(nn.Module):
    def forward(self, input):
        return swish(input)

