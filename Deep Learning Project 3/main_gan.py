# -*- coding: utf-8 -*-
"""
Created on Sun May 14 23:47:57 2023

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
from model_gan import Generator, Discriminator, Critic, initialize_weights
import PIL
from sklearn.manifold import TSNE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
z_dim = 100

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))]
)

train_dataset = torchvision.datasets.MNIST(root='./data', train = True, transform = transform, download = True)


def train():
    disc = Discriminator().to(device)
    gen = Generator(z_dim).to(device)
    
    initialize_weights(disc)
    initialize_weights(gen)
    
    gen.train()
    disc.train()
    
    num_epochs = 200
    batch_size = 128
    learning_rate_disc = 2e-4
    learning_rate_gen = 2e-4
    
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    
    optimizer_disc = torch.optim.Adam(disc.parameters(), lr = learning_rate_disc, betas = (0.5,0.999))      
    optimizer_gen = torch.optim.Adam(gen.parameters(), lr = learning_rate_gen, betas = (0.5,0.999))  
    criterion = nn.BCELoss()
    
    Disc_loss_values = []
    Gen_loss_values = []
    for epoch in range(num_epochs):
        
        Gen_loss = 0
        Disc_fake_loss = 0
        Disc_real_loss = 0
        for i, (real, labels) in enumerate(train_loader):  
            real = real.to(device)
            noise = torch.randn((real.size(0), z_dim, 1, 1)).to(device)
            fake = gen(noise)
            ## Discriminator Loss :  max log(D(x))+log(1-D(G(z))) or equivalently min -(log(D(x))+log(1-D(G(z))))
            
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real)) # -log(D(x))
            
            disc_fake = disc(fake).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) # -log(1-D(G(z)))
           
            loss_disc = (loss_disc_real + loss_disc_fake)/2
            
            disc.zero_grad()
            loss_disc.backward(retain_graph = True)  ## So that we dont lose the calculated "fake"
            optimizer_disc.step()
            
            ##  Generator Loss : min log(1-D(G(z))) but this way gradients saturate so I use the equivalent min -log(D(G(z)))
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output)) # -log(D(G(z)))
            
            gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()
            Gen_loss += loss_gen.item()*real.size(0)
            Disc_fake_loss += loss_disc_fake.item()*real.size(0)
            Disc_real_loss += loss_disc_real.item()*real.size(0)
            
        print("Generator loss =", Gen_loss/len(train_dataset),"Discriminator fake loss =", (Disc_fake_loss)/(len(train_dataset)),
              "Discrimnator real loss =", (Disc_real_loss)/len(train_dataset),"epoch =", epoch+1)
        Disc_loss_values.append((Disc_fake_loss+Disc_real_loss)/(2*len(train_dataset)))
        Gen_loss_values.append(Gen_loss/len(train_dataset))
        #filename = open("GAN_last"+str(epoch)+".pk",'wb')
        #pickle.dump(gen,filename)  
        #filename.close()
        
    x = np.arange(0, num_epochs, step=1)
    y_gen = np.array(Gen_loss_values)
    y_disc = np.array(Disc_loss_values)
    plt.plot(x, y_gen, label = "Generator")
    plt.plot(x, y_disc, label = "Discriminator")
    plt.legend()
    plt.show()

def gradient_penalty(critic, real, fake, device): # From WGAN-GP paper
    
    Batch_Size, c, h, w = real.shape
    epsilon = torch.rand((Batch_Size,1,1,1)).repeat(1, c, h, w).to(device)
    interpolated_images = real *epsilon + fake*(1-epsilon)
    
    score = critic(interpolated_images)
    
    gradient = torch.autograd.grad(
        inputs = interpolated_images,
        outputs = score,
        grad_outputs = torch.ones_like(score),
        create_graph = True,
        retain_graph = True,
        )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim = 1)   #L-2 Norm of the gradient
    gradient_penalty = torch.mean((gradient_norm -1)**2)
    
    return gradient_penalty
    
    
def Wasserstein_train():  # Implementation WGAN with gradient penalty
    
    gen = Generator(z_dim).to(device)
    critic= Critic().to(device)
    
    initialize_weights(critic)
    initialize_weights(gen)
    
    critic.train()
    gen.train()

    num_epochs = 200
    batch_size = 64
    learning_rate = 1e-4
    critic_updates = 2  # We update critic 2 times before updating generator
    Lambda = 10  # Regularization parameter used with gradient penalty
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr = learning_rate, betas = (0,0.999))      
    optimizer_gen = torch.optim.Adam(gen.parameters(), lr = learning_rate, betas = (0,0.999))  
    
    gen_loss_values = []
    critic_loss_values = []
    for epoch in range(num_epochs):
        
        Gen_loss = 0
        critic_loss = 0
        for i, (real, labels) in enumerate(train_loader):  
            real = real.to(device)
            
            noise = torch.randn((real.size(0), z_dim, 1, 1)).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device)
            # Critic wants to give high scores to real and low scores to fakes
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + Lambda*gp  
            critic.zero_grad()
            loss_critic.backward(retain_graph = True)
            optimizer_critic.step()
            critic_loss += loss_critic.item()*real.size(0)
                
            if(i%critic_updates == critic_updates-1):
                output = critic(fake).reshape(-1)
                loss_gen = -torch.mean(output)   # Generator wants critic to give high scores to its creations
                gen.zero_grad()
                loss_gen.backward()
                optimizer_gen.step()
                Gen_loss += loss_gen.item()*real.size(0)
           
           
        print("Generator loss =", critic_updates*Gen_loss/len(train_dataset), "Critic loss =", critic_loss/(len(train_dataset)),"epoch =", epoch+1)
        gen_loss_values.append(critic_updates*Gen_loss/len(train_dataset))
        critic_loss_values.append(critic_loss/len(train_dataset))
        
        #filename = open("WGAN_try"+str(epoch)+".pk",'wb')
        #pickle.dump(gen, filename)  
        #filename.close()
    x = np.arange(0, num_epochs, step=1)
    y_gen = np.array(gen_loss_values)
    y_critic = np.array(critic_loss_values)
    plt.plot(x, y_gen, label = "Generator")
    plt.plot(x, y_critic, label = "Critic")
    plt.legend()
    plt.show()

## To train Wasserstein GAN please comment out train() and uncomment Wasserstein_train() below.

#train()
Wasserstein_train()
