# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 03:46:55 2023

@author: USER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle



class CNN3(nn.Module):
    def __init__(self):
        super(CNN3,self).__init__()
        self.conv1=nn.Conv2d(3,128,5,padding=0)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(128,256,5,padding=1)
        self.conv3=nn.Conv2d(256,512,5,padding=1)
        self.fc1=nn.Linear(512*2*2,128)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2=nn.Dropout(0.15)
        self.dropout3=nn.Dropout(0.2)
        self.dropout4=nn.Dropout(0.5)
        self.fc2=nn.Linear(128,10)
        self.batchnorm1= nn.BatchNorm2d(128)
        self.batchnorm2= nn.BatchNorm2d(256)
        self.batchnorm3= nn.BatchNorm2d(512)
        self.batchnorm4= nn.BatchNorm1d(128)
        
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.batchnorm1(x)
        x=self.pool(x)
        x=self.dropout1(x)
        x=F.relu(self.conv2(x))
        x=self.batchnorm2(x)
        x=self.pool(x)
        x=self.dropout2(x)
        x=F.relu(self.conv3(x))
        y=self.pool(x)
        y=y.reshape(-1,512*2*2)
        x=self.batchnorm3(x)
        x=self.pool(x)
        x=x.reshape(-1,512*2*2)
        x=self.dropout3(x)
        x=F.relu(self.fc1(x))
        x=self.batchnorm4(x)
        x=self.dropout4(x)
        x=self.fc2(x)
        
        return [x,y]
    
    
    
    





        
        
        
        
        
        
        
        
        
        
 
    
 
    
 
    
 
    
 
    