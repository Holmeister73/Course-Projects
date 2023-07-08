# -*- coding: utf-8 -*-
"""
Created on Sun May 14 23:47:13 2023

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
import PIL
from sklearn.manifold import TSNE


input_size = 28
sequence_length = 28
hidden_size = 128


class VAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.LSTM_Encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        
        self.z_mean= nn.Linear(hidden_size, z_dim)
                    
        self.z_log_var= nn.Linear(hidden_size, z_dim)  # I used log of variance so that it can be both positive and negative which is better for training
                   
            
        
        self.Convolutional_Decoder=nn.Sequential(
            
                nn.ConvTranspose2d(z_dim, 256, stride = 1, kernel_size = 4, padding = 0),  # 256 tane 4*4
                nn.ReLU(inplace = True),
                nn.BatchNorm2d(256),
                nn.ConvTranspose2d(256, 128, stride = 2, kernel_size = 3, padding = 1), # 128 tnae 7*7 
                nn.ReLU(inplace = True),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(128, 64, stride = 2, kernel_size = 4, padding = 1), # 64 tane 14*14
                nn.ReLU(inplace = True),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 1, stride = 2, kernel_size = 4, padding = 1),  # 1 tane 28*28
                nn.Sigmoid()
            )
    
    def forward(self,x):
        x = self.LSTM_Encoder(x)[0][:,-1,:]  # I use the output of the last layer which is a 128 dimensional vector
        
        mu = self.z_mean(x)
        log_var = self.z_log_var(x)
        
        epsilon = torch.randn_like(log_var)
        z = mu + epsilon*torch.exp(log_var/2)
        encoded = z
        z = z.view(-1, self.z_dim, 1, 1) # to use it in convolutional decoder we turn 100 dimensional vector to 100, 1, 1 tensor
        
        x_hat = self.Convolutional_Decoder(z)
        return x_hat, mu, log_var, encoded
        