from sklearn import datasets, model_selection
data = datasets.load_iris()
Xtrain, Xtest, ytrain, ytest = \
    model_selection.train_test_split(data['data'], data['target']) 

#Inizializzazione
import numpy as np

import torch

Xtrain = torch.from_numpy(Xtrain).float()
Xtest = torch.from_numpy(Xtest).float()
ytrain = torch.from_numpy(ytrain)
ytest = torch.from_numpy(ytest)

# 1. Creazione di una rete neurale

# Inizializzazione del modello
lin = nn.Linear(4, 3)

# Predizione
lin(Variable(Xtrain[0:1]))

# class CustomModel(nn.Module):

#    def __init__(self):
#        # Codice per l'inizializzazione
#        pass

#    def forward(self, x):
#        # Codice per la forward pass
#        pass

class CustomModel(nn.Module):

    def __init__(self):
        super(CustomModel, self).__init__()

        self.hidden = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(10, 3)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        return self.out(self.drop(x))

net = CustomModel()
net(Variable(Xtrain[0:1]))

print(net)
# CustomModel(
# (hidden): Linear(in_features=4, out_features=10, bias=True)
# (relu): ReLU()
# (drop): Dropout(p=0.2)
# (out): Linear(in_features=10, out_features=3, bias=True)
# )

params = list(net.parameters())
len(params) 
# Prints: 4

print(params[-1])
# Parameter containing:
# -0.2020 -0.0427 0.2549 [torch.FloatTensor of size 3]

print(sum([torch.numel(p) for p in params])) 
# Prints: 83

named_params = [p for p in net.named_parameters()]
print(named_params[-1])
# (
# 'out.bias', 
# Parameter containing:
# -0.2020 -0.0427 0.2549 [torch.FloatTensor of size 3]
# )

import torch.nn.functional as F

def logreg(x):
  return F.softmax(lin(x), dim=1)

relu_inplace = nn.ReLU(inplace=True)

net_sequential = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(10, 3)
)

torch.nn.init.normal(net.hidden.weight.data)
torch.nn.init.constant(net.hidden.bias.data, 0.1)

torch.nn.init.normal_(net.hidden.weight)
torch.nn.init.constant_(net.hidden.bias, 0.1)

for m in net.modules():
  if type(m) in ['Linear']:
    torch.nn.init.normal(m.weight.data)
    torch.nn.init.constant(m.bias.data, 0.1)

# 2. Ottimizzazione
net = CustomModel()
loss = nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=net.parameters(), lr=0.01)

Xt = Variable(Xtrain)
yt = Variable(ytrain)

def train_step(x, y):

  # Modalità di training
  net.train()

  # Calcola le predizioni
  y_pred = net(x)

  # Calcola funzione costo
  loss_epoch = loss(y_pred, y)

  # Esegui back-propagation
  loss_epoch.backward()

  # Aggiorna le variabili
  opt.step()

  # Resetta il gradiente
  opt.zero_grad()

for epoch in range(2500):
  train_step(Xt, yt)

def accuracy(y_pred, y_true):
  correct = (y_pred.max(dim=1)[1] == y_true)
  return torch.mean(correct.float()).data.numpy()

# 3. Gestione dei dati con il modulo data
from torch.utils import data
train_data = data.TensorDataset(Xtrain, ytrain)

print(train_data[0]) 
# Prints: (4.9000 2.5000 4.5000 1.7000 [torch.FloatTensor of size 4], 2)

print(len(train_data))
# Prints: 112

train_data_loader = data.DataLoader(train_data, batch_size=32, shuffle=True)

for epoch in range(1000):
  net.train()
  for Xb, yb in train_data_loader:
    train_step(Variable(Xb), Variable(yb))

# 4. Ottimizzazione su GPU invece che su CPU
torch.cuda.device_count()            # Numero di GPU disponibili
torch.cuda.get_device_name(0)        # Nome della prima GPU disponibile
torch.cuda.current_device()          # Device in uso al momento
torch.cuda.set_device(0)             # Imposta la prima GPU come default
torch.cuda.get_device_capability(0)  # Verifica le capacità della prima GPU

net = CustomModel()
if torch.cuda.is_available():
  net.cuda()
loss = nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=net.parameters(), lr=0.01)

device = torch.device("cuda" if use_cuda else "cpu")
net.to(device)

for epoch in range(2500):
  net.train()
  for Xb, yb in train_data_loader:

    if torch.cuda.is_available():
      Xb, yb = Variable(Xb).cuda(), Variable(yb).cuda()
    else:
      Xb, yb = Variable(Xb), Variable(yb)

    train_step(Xb, yb)

net.cpu()

# 5. Checkpointing del modello
torch.save(net.state_dict(), './tmp')

net.load_state_dict(torch.load('./tmp'))

start_epoch = resume_from_checkpoint('checkpoint.pth.tar')
for epoch in range(start_epoch, 1000):

  net.train()

  for Xb, yb in train_data_loader:
    Xb, yb = Variable(Xb), Variable(yb)

    train_step(Xb, yb)

  # Stato complessivo del processo di ottimizzazione
  state = {
    'epoch': epoch,
    'state_dict': net.state_dict(),
    'opt': opt.state_dict(),
  }
  torch.save(state, 'checkpoint.pth.tar') 

import os
def resume_from_checkpoint(path_to_checkpoint):

  if os.path.isfile(path_to_checkpoint):

    # Caricamento del checkpoint
    checkpoint = torch.load(path_to_checkpoint)

    # Ripristino dello stato del sistema
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    opt.load_state_dict(checkpoint['opt'])
    print("Caricato il checkpoint '{}' (epoca {})"
                  .format(path_to_checkpoint, checkpoint['epoch']))

  else:
    start_epoch = 0

  return start_epoch

