#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:52:26 2021

@author: tathagat
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import syft as sy
import copy
import numpy as np
import time

import importlib
importlib.import_module('NonIID')
from NonIID import load_dataset, getActualImages
from utils import averageModels, averageGradients

'''
Create a list of clients, where each client will be a dictionary. The dictionaries will have DataLoader, Model etc.
'''

#Hyperparameters

class Arguments():
    def __init__(self):
        self.images = 60000        #Training 
        self.clients = 10          #Number of clients
        self.rounds = 5            #From global to local, and then local to global.
        self.epochs = 5            #Epochs to train model on each client
        self.local_batches = 64     
        self.lr = 0.01             #Learning rate
        self.C = 0.9               #Random fraction to include the number of clients
        self.drop_rate = 0.1       #To calculate number of active devices are
        self.torch_seed = 0        #To have the exact same weights and parameters while re-running the code
        self.log_interval = 10     #Loss interval
        self.iid = 'iid'           #Type of data
        self.split_size = int(self.images / self.clients)
        self.samples = self.split_size / self.images 
        self.use_cuda = False
        self.save_model = False
        
args = Arguments()

use_cuda = args.use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

#Creating virtual workers
hook = sy.TorchHook(torch)
clients = []

for i in range(args.clients):
    clients.append({'hook': sy.VirtualWorker(hook, id="client{}".format(i+1))})

#load_dataset to import the IID data
global_train, global_test, train_group, test_group = load_dataset(args.clients, args.iid)


#Return actual images and labels for indicies
for inx, client in enumerate(clients):
    trainset_ind_list = list(train_group[inx])
    client['trainset'] = getActualImages(global_train, trainset_ind_list, args.local_batches)
    client['testset'] = getActualImages(global_test, list(test_group[inx]), args.local_batches)
    client['samples'] = len(trainset_ind_list) / args.images
    

#Load test dataset for global model
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
global_test_dataset = datasets.MNIST('./', train=False, download=True, transform=transform)
global_test_loader = DataLoader(global_test_dataset, batch_size=args.local_batches, shuffle=True)


#Class for our network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def ClientUpdate(args, device, client):
    client['model'].train()
    client['model'].send(client['hook'])
    
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (data, target) in enumerate(client['trainset']):
            data = data.send(client['hook'])
            target = target.send(client['hook'])
            
            data, target = data.to(device), target.to(device)
            client['optim'].zero_grad()
            output = client['model'](data)
            loss = F.nll_loss(output, target)
            loss.backward()
            client['optim'].step()
            
            if batch_idx % args.log_interval == 0:
                loss = loss.get() 
                print('Model {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    client['hook'].id,
                    epoch, batch_idx * args.local_batches, len(client['trainset']) * args.local_batches, 
                    100. * batch_idx / len(client['trainset']), loss))
                
    client['model'].get()


def test(args, model, device, test_loader, name):
    model.eval()   
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss for {} model: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        name, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    


torch.manual_seed(args.torch_seed)
global_model = Net()

for client in clients:
    torch.manual_seed(args.torch_seed)
    client['model'] = Net().to(device)
    client['optim'] = optim.SGD(client['model'].parameters(), lr=args.lr)

for fed_round in range(args.rounds):
    
#     uncomment if you want a randome fraction for C every round, otherwise C = 0.9 as defined earlier
#     args.C = float(format(np.random.random(), '.1f'))
    
    # number of selected clients
    m = int(max(args.C * args.clients, 1))

    # Selected devices
    np.random.seed(fed_round)
    selected_clients_inds = np.random.choice(range(len(clients)), m, replace=False)
    selected_clients = [clients[i] for i in selected_clients_inds]
    
    # Active devices
    np.random.seed(fed_round)
    active_clients_inds = np.random.choice(selected_clients_inds, int((1-args.drop_rate) * m), replace=False)
    active_clients = [clients[i] for i in active_clients_inds]
    
    # Training 
    for client in active_clients:
        ClientUpdate(args, device, client)
    
#     # Testing 
#     for client in active_clients:
#         test(args, client['model'], device, client['testset'], client['hook'].id)
    
    # Averaging 
    global_model = averageModels(global_model, active_clients)
    
    # Testing the average model
    test(args, global_model, device, global_test_loader, 'Global')
            
    # Share the global model with the clients
    for client in clients:
        client['model'].load_state_dict(global_model.state_dict())
        
if (args.save_model):
    torch.save(global_model.state_dict(), "FedAvg.py")


