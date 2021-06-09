#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 23:43:47 2021

@author: tathagat
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

"""
Consider we have 60k images for training and 10 users.
Number of images per user will be 6k. 
Inidices for the images will be 0 to 5999 (here).
We have to ensure that the same images are not repeated for different users while training.
We want to append the user wise data into a single DS. We don't want multiple DS for
the same. So, we create a user dictionary to store the list of image indices for different users.
"""
    
def mnistIID(dataset, num_users):
    num_images = int(len(dataset)/num_users)
    users_dict, indices = {}, list(range(len(dataset)))
    for i in range(num_users):
        np.random.seed(i) #Imp to have same weight values everytime we rerun the code
        #set() by default drops the repeated items (images, here)
        users_dict[i] = set(np.random.choice(indices, num_images, replace = False))
        indices = list(set(indices) - users_dict[i]) #So that same images are not repeated for different users
    return users_dict
    

#Importing train, test datasets and train, test groups 
def load_dataset(num_users):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    train_dataset = datasets.MNIST(root = './data', train = True, transform = transforms.Compose([transforms.ToTensor()]), download = True)
    test_dataset = datasets.MNIST(root = './data', train = False, transform = transforms.Compose([transforms.ToTensor()]), download = True)
    train_group = mnistIID(train_dataset, num_users)
    test_group = mnistIID(test_dataset, num_users)
    return train_dataset, test_dataset, train_group, test_group

class FedDataset(Dataset):
    def __init__(self, dataset, indx):
        self.dataset = dataset
        self.indx = [int(i) for i in indx]
        
    def __len__(self):
        return len(self.indx)
    
    def __getitem__(self, item):
        images, label = self.dataset[self.indx[item]]
        return torch.tensor(images), torch.tensor(label)

#This function takes the dataset and the list of indices
def getActualImages(dataset, indices, batch_size):
    return DataLoader(FedDataset(dataset, indices), batch_size = batch_size, shuffle = True)




    
    
