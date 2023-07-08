# -*- coding: utf-8 -*-
"""
Created on Sun May 14 23:47:25 2023

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





class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            GaussianNoise(0.1), # I used Gaussian noise in discriminator to make sure it is not that confident from the start otherwise gradient
            # signals for generator would be too low
            nn.Conv2d(1, 32, kernel_size = 4, stride = 2, padding = 1),   # 28*28 -> 14*14
            nn.LeakyReLU(0.2, inplace = True),   
            #nn.BatchNorm2d(32), #I didn't use batchnorm in this layer following the dcgan paper
            nn.Dropout(0.15),
            GaussianNoise(0.1),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1), #  14*14 -> 7*7
            nn.LeakyReLU(0.2, inplace = True),  
            nn.BatchNorm2d(64),  
            nn.Dropout(0.15),
            GaussianNoise(0.1),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),#   7*7 --> 4*4
            nn.LeakyReLU(0.2, inplace = True),
            nn.BatchNorm2d(128),
            nn.Dropout(0.15),
            GaussianNoise(0.1),
            nn.Conv2d(128, 1, kernel_size = 4, stride = 1, padding = 0), #4*4 --> 1*1
            nn.Sigmoid()
        )
        
    def forward(self,x):
        x = self.disc(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        
        self.gen = nn.Sequential(
                    nn.ConvTranspose2d(z_dim, 128, stride = 1, kernel_size = 4, padding = 0), # 1*1 --> 4*4
                    nn.ReLU(inplace = True),
                    nn.BatchNorm2d(128),
                    nn.ConvTranspose2d(128, 64, stride = 2, kernel_size = 3, padding = 1), # 4*4 --> 7*7
                    nn.ReLU(inplace = True),
                    nn.BatchNorm2d(64),
                    nn.ConvTranspose2d(64, 32, stride = 2, kernel_size = 4, padding = 1), # 7*7 --> 14*14
                    nn.ReLU(inplace = True),
                    nn.BatchNorm2d(32),
                    nn.ConvTranspose2d(32, 1, stride = 2, kernel_size = 4, padding = 1), # 14*14 --> 28*28
                    nn.Tanh()
                )
            
    def forward(self,x):
        generated = self.gen(x)
        return generated


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            GaussianNoise(0.1),  
            nn.Conv2d(1, 32, kernel_size = 4, stride = 2, padding = 1),   # 28*28 -> 14*14
            nn.LeakyReLU(0.2, inplace = True),   
            nn.Dropout(0.05),
            GaussianNoise(0.1),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1), #  14*14 -> 7*7
            nn.LeakyReLU(0.2, inplace = True),  
            nn.InstanceNorm2d(64),  
            nn.Dropout(0.05),
            GaussianNoise(0.1),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),#   7*7 --> 4*4
            nn.LeakyReLU(0.2, inplace = True),
            nn.InstanceNorm2d(128),
            nn.Dropout(0.05),
            GaussianNoise(0.1),
            nn.Conv2d(128, 1, kernel_size = 4, stride = 1, padding = 0), #4*4 --> 1*1
        )
        
    def forward(self,x):
        x = self.disc(x)
        return x
    
class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, activation):
        if self.training:
            return activation + torch.autograd.Variable(torch.randn(activation.size()).cuda() * self.stddev)
        return activation
     
def initialize_weights(model):       # This is again from DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)
        
        