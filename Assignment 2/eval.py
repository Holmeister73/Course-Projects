# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 03:47:15 2023

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
from model import CNN3


device= torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size=8

transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])

train_data=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)

test_data=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)

train_loader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=False)

test_loader=torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=False)

def evaluate(network):
    with torch.no_grad():
       network.eval()
       samples=0
       correct=0
      
       for j,(images,labels) in enumerate(test_loader):
           images = images.to(device)
           imagesFLIP = transforms.functional.hflip(images)
           imagesFLIP = imagesFLIP.to(device)
           labels = labels.to(device)
           outputs = network(images)[0]
           
           outputsFLIP = network(imagesFLIP)[0]
         
           finaloutput = (outputs+outputsFLIP)/2
           _,predicted = torch.max(finaloutput,1)
           
           samples += labels.size(0)
           correct += (predicted==labels).sum().item()
               
       acc = correct/samples
       return acc
   
def min_max_scaler(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

class_colors={"plane":"cyan","car":"orange","bird":"green","cat":"red","deer":"purple",
              "dog":"brown","frog":"pink","horse":"black","ship":"gray","truck":"blue"}

def Visualize(network):
    with torch.no_grad():
        network.eval()
        Features=[]
        Label_index=[[],[],[],[],[],[],[],[],[],[]]
        for i,(images,labels) in enumerate(train_loader):
            images=images.to(device)
            labels=labels.to(device)
            for j in range(images.size(0)):
                Label_index[labels[j].item()].append(8*i+j)
                Features.append(network(images)[1][j].cpu().numpy())
        Features=np.array(Features)
        print(Features.shape)
        tsne = TSNE(n_components=2).fit_transform(Features)
        
        plt.figure(figsize = (20,15))
        x=min_max_scaler(tsne[:,0])
        y=min_max_scaler(tsne[:,1])
        for k,label in enumerate(class_colors):
            label_indices=Label_index[k]
            print(len(label_indices))
            label_x=np.take(x,label_indices)
            label_y=np.take(y,label_indices)
            color=class_colors[label]
            plt.scatter(label_x,label_y,c=color,label=label)
        
        plt.legend(loc="best")
        plt.show()




## Below are example codes for visualizing and evaluating a network.
network290=pickle.load(open("network290.pk","rb"))

#network0=pickle.load(open("network0.pk","rb"))

#network145=pickle.load(open("network145.pk","rb"))
#Visualize(network0)
#Visualize(network145)
#Visualize(network290)
print(evaluate(network290))       