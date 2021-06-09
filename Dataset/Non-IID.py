#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 02:15:32 2021

@author: tathagat
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from IID_Functions import mnistIID

"""
Distribute the data on the basis of number classes.
So if we average the local model of IID data, it will be better than the 
non-IID dataset. The reason being local models must have seen more classes in
IID distribution of data.
vstack take two numpy arrays and stack them together (column wise) and then sort.
In this way we won't lose the indices.
"""


def mnistNonIID(dataset, num_users):
    classes, images = 100, 600
    class_indx = [i for i in range(clasess)]
    users_dict - {i: np.array([]) for i in range(num_users)}
    indices = np.arrange(classes*images)  #Length of dataset i.e 60k
    unsorted_labels = dataset.train_labels.numpy()
    
    #The following list consists of indices and the unsorted labels
    indices_unlabels = np.vstack((indices, unsorted_labels))
    labels = indices_unlabels[:, indices_unlabels[1,:].argsort()]
    indices = labels[0, :]
    
    for i in range(num_users):
        temp = set(np.random.choice(classes_indx, 2, replace = False))
        classes_indx = list(set(classes_indx) - temp)
        for t in temp:
            users_dict[i] = np.concatenate((users_dict[i], indices[t*images:(t+1)*images]), axis=0)
    return users_dict

def mnistNonIID_Unequal(dataset, num_users):
    classes, images = 1200, 50
    if test:
        classes, images = 200, 50
    classes_indx = [i for i in range(classes)]
    users_dict = {i: np.array([]) for i in range(num_users)}
    indeces = np.arange(classes*images)
    unsorted_labels = dataset.train_labels.numpy()

    indeces_unsortedlabels = np.vstack((indeces, unsorted_labels))
    indeces_labels = indeces_unsortedlabels[:, indeces_unsortedlabels[1, :].argsort()]
    indeces = indeces_labels[0, :]

    min_cls_per_client = 1
    max_cls_per_client = 30

    random_selected_classes = np.random.randint(min_cls_per_client, max_cls_per_client+1, size=num_users)
    random_selected_classes = np.around(random_selected_classes / sum(random_selected_classes) * classes)
    random_selected_classes = random_selected_classes.astype(int)

    if sum(random_selected_classes) > classes:
        for i in range(num_users):
            np.random.seed(i)
            temp = set(np.random.choice(classes_indx, 1, replace=False))
            classes_indx = list(set(classes_indx) - temp)
            for t in temp:
                users_dict[i] = np.concatenate((users_dict[i], indices[t*images:(t+1)*images]), axis=0)

        random_selected_classes = random_selected_classes-1

        for i in range(num_users):
            if len(classes_indx) == 0:
                continue
            class_size = random_selected_classes[i]
            if class_size > len(classes_indx):
                class_size = len(classes_indx)
            np.random.seed(i)
            temp = set(np.random.choice(classes_indx, class_size, replace=False))
            classes_indx = list(set(classes_indx) - temp)
            for t in temp:
                users_dict[i] = np.concatenate((users_dict[i], indices[t*images:(t+1)*images]), axis=0)
    else:

        for i in range(num_users):
            class_size = random_selected_classes[i]
            np.random.seed(i)
            temp = set(np.random.choice(classes_indx, class_size, replace=False))
            classes_indx = list(set(classes_indx) - temp)
            for t in temp:
                users_dict[i] = np.concatenate((users_dict[i], indices[t*images:(t+1)*images]), axis=0)

        if len(classes_indx) > 0:
            class_size = len(classes_indx)
            j = min(users_dict, key=lambda x: len(users_dict.get(x)))
            temp = set(np.random.choice(classes_indx, class_size, replace=False))
            classes_indx = list(set(classes_indx) - temp)
            for t in temp:
                users_dict[j] = np.concatenate((users_dict[j], indices[t*images:(t+1)*images]), axis=0)

    return users_dict


def load_dataset(num_users, iidtype):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    train_dataset = datasets.MNIST(root = './data', train = True, transform = transforms.Compose([transforms.ToTensor()]), download = True)
    test_dataset = datasets.MNIST(root = './data', train = False, transform = transforms.Compose([transforms.ToTensor()]), download = True)
    train_group, test_group = None, None
    
    if iidtype == 'iid':
        train_group = mnistIID(train_dataset, num_users)
        test_group = mnistIID(test_dataset, num_users)
        
    elif iidtype == 'noniid':
        train_group = mnistNonIID(train_dataset, num_users)
        test_group = mnistNonIID(test_dataset, num_users)
        
    else:
        train_group = mnistNonIID_Unequal(train_dataset, num_users)
        test_group = mnistNonIID_Unequal(test_dataset, num_users)
    
    
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
    
def getActualImages(dataset, indices, batch_size):
    return DataLoader(FedDataset(dataset, indices), batch_size = batch_size, shuffle = True)


random_selected_classes = np.random.randint(1, 11, size = 10)
print(random_selected_classes)
print(sum(random_selected_classes))

random_selected_classes = np.around(random_selected_classes / sum(random_selected_classes) * 100) #Number of classes = 100
print(random_selected_classes)

random_selected_classes = random_selected_classes.astype(int)
print(random_selected_classes)
print(sum(random_selected_classes.astype(int)))

