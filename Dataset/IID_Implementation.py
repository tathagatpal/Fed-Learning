#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 00:16:03 2021

@author: tathagat
"""

from IID_Functions import load_dataset, getActualImages

train_dataset, test_dataset, train_group, test_group = load_dataset(10) #Users = 10

first_client_batches = getActualImages(train_dataset, train_group[0], 64)

for images, labels in first_client_batches:
    print(len(images))
    break
