# -*- coding: utf-8 -*-
"""
Created on Sun May 14 23:47:45 2023

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
from model_vae import VAE


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 200
batch_size = 256
learning_rate = 1e-3
Lambda = 0.0025
z_dim = 100

train_dataset = torchvision.datasets.MNIST(root='./data', train = True, transform = transforms.ToTensor(),  download = True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

vae = VAE(z_dim).to(device)

vae.train()

optimizer = torch.optim.Adam(vae.parameters(), lr = learning_rate)  
reconstruction_criterion = nn.BCELoss()



Reconstruction_loss_values = []
KL_divergence_loss_values = []
Total_loss_values = []

for epoch in range(num_epochs):
    Total_loss = 0
    Reconstruction_loss = 0
    KL_div_loss = 0
    for i, (images, labels) in enumerate(train_loader):  
       
        
        images = images.to(device)
        images_to_feed = images.reshape(-1,28,28).to(device) # Reshape images from 1x28x28 to 28x28 to use in LSTM
   
    
        reconstructed, mean, log_var, encoded = vae(images_to_feed)
        
        standard_normal = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(100).to(device),covariance_matrix=torch.eye(100).to(device))
        
        var = (torch.exp(log_var)).to(device)
        covar = torch.diag_embed(var, dim1 = 1, dim2 = 2).to(device)
        encoded_normal=torch.distributions.multivariate_normal.MultivariateNormal(loc = mean, covariance_matrix = covar)
        
        kl_divergence_loss = torch.distributions.kl.kl_divergence(encoded_normal, standard_normal).mean()
        
        reconstruction_loss = reconstruction_criterion(reconstructed, images)
        loss = reconstruction_loss + Lambda*kl_divergence_loss
        
        Total_loss+=loss.item()*images.size(0)
        Reconstruction_loss+=reconstruction_loss.item()*images.size(0)
        KL_div_loss+=kl_divergence_loss.item()*images.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
        
    print("Total loss:",Total_loss/len(train_dataset),"Reconstruction loss:", Reconstruction_loss/len(train_dataset),
          "KL loss:", KL_div_loss/len(train_dataset),"epoch =", epoch+1) 
    Reconstruction_loss_values.append(Reconstruction_loss/len(train_dataset))
    KL_divergence_loss_values.append(Lambda*KL_div_loss/len(train_dataset))
    Total_loss_values.append(Total_loss/len(train_dataset))
    #filename = open("VAE"+str(epoch)+".pk",'wb')
    #pickle.dump(vae, filename) 
    #filename.close()
    
x = np.arange(0, num_epochs, step=1)
y_reconstruction = np.array(Reconstruction_loss_values)
y_kldiv = np.array(KL_divergence_loss_values)
y_total = np.array(Total_loss_values)
plt.plot(x, y_reconstruction, label = "Reconstruction")
plt.plot(x, y_kldiv, label = "KL divergence")
plt.plot(x, y_total, label = "Total")
plt.legend()
plt.show()
        
