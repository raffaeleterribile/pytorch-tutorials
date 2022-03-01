from torchvision import datasets
fmnist = datasets.FashionMNIST('fmnist', train=True, download=True)

fmnist_test = datasets.FashionMNIST('fmnist', train=False)

print(len(fmnist))
# Ritorna: 60000
print(fmnist[0])
# Ritorna: (<PIL.Image.Image image mode=L size=28x28 at 0x7F20B7393048>, tensor(9))

from torchvision import transforms
fmnist = datasets.FashionMNIST('fmnist', transform=transforms.ToTensor())

# Creiamo un loader per mini-batch di sedici immagini
from torch.utils import data
data_loader = data.DataLoader(fmnist, batch_size=16)

# Prendiamo il primo mini-batch
xb, yb = next(iter(data_loader))

# Lo stampiamo su schermo
from torchvision import utils
import matplotlib.pyplot as plt
out = torchvision.utils.make_grid(xb)
plt.imshow(out.numpy().transpose((1, 2, 0)))

# Dobbia trasformazione
tr = transforms.Compose([
    transforms.RandomRotation(degrees=75),
    transforms.ToTensor()
])
fmnist = datasets.FashionMNIST('fmnist', train=True, transform=tr) 

tr = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x < 0.5).float())
])
fmnist = torchvision.datasets.FashionMNIST('fmnist', transform=tr)

# Calcola media e variazione standard
import numpy as np
im_mean = np.mean(fmnist.train_data.numpy())
im_std = np.std(fmnist.train_data.numpy())

# Applica la normalizzazione
tr = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((im_mean / 255.0,), (im_std / 255.0,))
])
fmnist = torchvision.datasets.FashionMNIST('fmnist', train=True, transform=tr)

from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

from torchvision import models
net = models.resnet18(pretrained=True)

from PIL import Image
import requests
im = Image.open(requests.get('https://www.ikea.com/us/en/images/products/ekedalen-chair-gray__0516596_PE640434_S4.JPG', stream=True).raw)

tr = transforms.Compose([
    # Ridimensiona a 224 x 224
    transforms.Resize(224),
    # Transforma in tensore
    transforms.ToTensor(),
    # Normalizza
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
im = tr(im)

net.eval()
torch.argmax(net(im.unsqueeze(0)))

net.fc = torch.nn.Linear(net.fc.in_features, 10)

print(sum(p.numel() for p in net.parameters() if p.requires_grad))
# Ritorna: 11689512

for param in net.parameters():
    param.requires_grad = False
net.fc = torch.nn.Linear(net.fc.in_features, 10)

