#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 00:25:13 2021

@author: tathagat
"""

import torch
import syft as sy

hook = sy.TorchHook(torch)

john = sy.VirtualWorker(hook, id = 'john') #A client named John

print(john._objects)

x= torch.Tensor([1,2,3])
x = x.send(john) #Actual tensor x is going to be sent at John

print(x) #x points to bob

print(john._objects)

print(x.location)

print(john.id)

print(x.location.id)

x = x.get() #Receive the actual tensor that was sent to John
