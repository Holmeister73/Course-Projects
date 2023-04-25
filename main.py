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

if(torch.cuda.is_available()):
    processor=torch.device("cuda")
    
else:
    processor=torch.device("cpu")


#TODO try adding residual connections
#TODO try adding regularization
print(processor)
learning_rate=0.001
batch_size=128
epoch_number=150
transform= transforms.Compose([
   
    #transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    #transforms.RandomGrayscale(p=0.2),
    #transforms.Resize((36,36)),
    #transforms.RandomCrop((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))]
    )

transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    
train_data=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
print(len(train_data))
test_data=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform2)
train_loader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
train_loader2=torch.utils.data.DataLoader(train_data,batch_size=2,shuffle=False) # for tsne
test_loader=torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=False)
#psum=torch.tensor([0.0,0.0,0.0])
#psum_sq=torch.tensor([0.0,0.0,0.0])
#for (images,labels) in train_loader2:
#    psum+=images.sum(axis=[0,2,3])
#    psum_sq+=(images**2).sum(axis=[0,2,3])

#count=len(train_data)*32*32
#total_mean=psum/count
#total_var=psum_sq/count-(total_mean)**2
#total_std=torch.sqrt(total_var)
#print(total_mean,total_std)

network=CNN3().to(processor)
loss_func=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(network.parameters(),lr=learning_rate,momentum=0.97)
optimizer2=torch.optim.Adam(network.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

def calculate_test_acc(network):
    network.eval()
    samples=0
    correct=0
    correct2=0
    correct3=0
    
    for j,(images,labels) in enumerate(test_loader):
        images=images.to(processor)
        imagesFLIP=transforms.functional.hflip(images)
        imagesAFFINE=transforms.RandomAffine(degrees=0,translate=(0.1,0.1))(images)
        imagesAFFINE=imagesAFFINE.to(processor)
        imagesFLIP=imagesFLIP.to(processor)
        labels=labels.to(processor)
        outputs=network(images)[0]
        
        outputsFLIP=network(imagesFLIP)[0]
        outputsAFFINE=network(imagesAFFINE)[0]
      
        finaloutput2=(outputs+outputsFLIP)/2
        finaloutput3=(outputs+outputsFLIP+outputsAFFINE)/3
        _,predicted=torch.max(outputs,1)
        _,predicted2=torch.max(finaloutput2,1)
        _,predicted3=torch.max(finaloutput3,1)
        #_,predictedFLIP=torch.max(outputsFLIP,1)
        # print(predicted)
        
        samples+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        correct2+=(predicted2==labels).sum().item()
        correct3+=(predicted3==labels).sum().item()
            
    acc=correct/samples
    acc2=correct2/samples
    acc3=correct3/samples
    return acc2

def calculate_train_acc(network):
    network.eval()
    samples=0
    correct=0
    
    for j,(images,labels) in enumerate(train_loader):
        images=images.to(processor)
        labels=labels.to(processor)
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
        images=images.to(processor)
        labels=labels.to(processor)
        output=network(images)
        predictions=output[0]
        loss=loss_func(predictions,labels)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        AverageLoss+=loss.item()*images.size(0)
    
    test_accuracy=calculate_test_acc(network)
    #train_accuracy=calculate_train_acc(network)
    print(test_accuracy,epoch)
    loss_values.append(AverageLoss/len(train_data))            
    filename=open("network"+str(epoch)+".pk",'wb')
    pickle.dump(network,filename)  
    
    print(epoch,AverageLoss/len(train_data))
        
 
print("Training done")
x=np.arange(0,epoch_number,step=1)
y=np.array(loss_values)
plt.plot(x,y)



