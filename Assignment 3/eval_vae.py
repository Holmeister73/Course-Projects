# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:31:09 2023

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
from sklearn.manifold import TSNE
from model_vae import VAE

batch_size=8
asd=182
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def min_max_scaler(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)


class_colors={"0":"cyan","1":"orange","2":"green","3":"red","4":"purple",
              "5":"brown","6":"pink","7":"black","8":"gray","9":"blue"}
def Visualize(network):
    with torch.no_grad():
        network.eval()
        Features=[]
        Label_index=[[],[],[],[],[],[],[],[],[],[]]
        for i,(images,labels) in enumerate(train_loader):
            images=images.to(device)
            images_to_feed = images.reshape(-1,28,28).to(device)
            labels=labels.to(device)
            outputs,mu,log_var,encoded = network(images_to_feed)
            for j in range(images.size(0)):
                Label_index[labels[j].item()].append(8*i+j)
                Features.append(encoded[j].cpu().numpy())
        Features=np.array(Features[:2000])
        print(Features.shape)
        tsne = TSNE(n_components=2).fit_transform(Features)
        
        plt.figure(figsize = (20,15))
        x=min_max_scaler(tsne[:,0])
        y=min_max_scaler(tsne[:,1])
        for k,label in enumerate(class_colors):
            label_indices=list(filter(lambda x: x<2000,Label_index[k]))
            print(len(label_indices))
            label_x=np.take(x,label_indices)
            label_y=np.take(y,label_indices)
            color=class_colors[label]
            plt.scatter(label_x,label_y,c=color,label=label)
        
        plt.legend(loc="best")
        plt.show()

network=pickle.load(open("VAE"+str(asd)+".pk","rb"))

Visualize(network)