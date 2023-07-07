# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 03:47:09 2023

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
from model import CNN3
import PIL
from sklearn.manifold import TSNE


device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate=0.001 #default for adam
batch_size=128
epoch_number=100


transform= transforms.Compose([
    transforms.ColorJitter(brightness=0.05,saturation=0.05,contrast=0.05),
    transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))]
    )

transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    
train_data=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)

test_data=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform2)

train_loader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)

test_loader=torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=False)

#psum=torch.tensor([0.0,0.0,0.0])
#psum_sq=torch.tensor([0.0,0.0,0.0])
#for (images,labels) in train_loader2:
#    psum+=images.sum(axis=[0,2,3])
#    psum_sq+=(images**2).sum(axis=[0,2,3])       These parts are for calculating mean and std of CIFAR 10 to use in standardizing
                                                    # results:  (0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)
#count=len(train_data)*32*32
#total_mean=psum/count
#total_var=psum_sq/count-(total_mean)**2
#total_std=torch.sqrt(total_var)
#print(total_mean,total_std)


network=CNN3().to(device)
loss_func=nn.CrossEntropyLoss()
optimizer0=torch.optim.SGD(network.parameters(),lr=0.1)
optimizer1=torch.optim.SGD(network.parameters(),lr=0.1,momentum=0.9)
optimizer2=torch.optim.Adam(network.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer3=torch.optim.AdamW(network.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01) #to test L_2 regularizer effect
optimizer4=torch.optim.Adagrad(network.parameters(),lr=0.01)
optimizer5=torch.optim.RMSprop(network.parameters(),lr=0.01)


def calculate_test_acc(network):
    with torch.no_grad():
        network.eval()
        samples=0
        correct=0
        
        for j,(images,labels) in enumerate(test_loader):
            images=images.to(device)
            imagesFLIP=transforms.functional.hflip(images)
            imagesFLIP=imagesFLIP.to(device)
            labels=labels.to(device)
            
            outputs=network(images)[0]
            outputsFLIP=network(imagesFLIP)[0]
            finaloutput=(outputs+outputsFLIP)/2
           
            _,predicted=torch.max(finaloutput,1)
            
            samples+=labels.size(0)
            correct+=(predicted==labels).sum().item()
                            
    acc=correct/samples
    
    return acc

def calculate_train_acc(network):
    with torch.no_grad():
        network.eval()
        samples=0
        correct=0
        
        for j,(images,labels) in enumerate(train_loader):
            images=images.to(device)
            labels=labels.to(device)
            outputs=network(images)[0]
            _,predicted=torch.max(outputs,1)
            samples+=labels.size(0)
            correct+=(predicted==labels).sum().item()
        
    acc=correct/samples
    return acc

loss_values=[]

for epoch in range(epoch_number):
    AverageLoss=0
    network.train()
    train_correct=0
    for i,(images,labels) in enumerate(train_loader):
        images=images.to(device)
        labels=labels.to(device)
        output=network(images)
        predictions=output[0]
        loss=loss_func(predictions,labels)
        optimizer5.zero_grad()
        loss.backward()
        optimizer5.step()
        AverageLoss+=loss.item()*images.size(0)
    
    test_accuracy=calculate_test_acc(network)
    #train_accuracy=calculate_train_acc(network)
    print(epoch,test_accuracy,AverageLoss/len(train_data))
    loss_values.append(AverageLoss/len(train_data))            
    #filename=open("networkC"+str(epoch)+".pk",'wb')
    #pickle.dump(network,filename)  
    
        
 
print("Training done")
x=np.arange(0,epoch_number,step=1)
y=np.array(loss_values)
plt.plot(x,y)



