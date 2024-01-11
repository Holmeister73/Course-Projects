# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 21:30:32 2024

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
from torch.utils.data import random_split
import torchvision.models as models
from model import ResNet34, ResNet34_Pencil
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from extra_datasets import Cub200Dataset, Dataset_with_Indices
from utils import one_hot_encode, margin_based_loss, soft_cross_entropy, cross_entropy_without_softmax, kl_loss, calculate_class_embeddings
from utils import entropy, calculate_valid_acc, pencil_valid_acc, symmetric_new_label, make_noisy, manual_cross_entropy
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torch.nn.utils import clip_grad_norm_ 
import gc
import math

def train_default(dataset = "cifar10", noise_level = 0, noise_type = "symmetric", batch_size = 128, lr = 0.1, momentum = 0.9, weight_decay = 5e-4, epoch_number = 200):
    
    if(dataset == "cifar10"):
        original_labels = cifar10_train.targets.copy()
        class_number = 10
        if(noise_level != 0):
            noisy_targets = make_noisy(cifar10_train.targets, class_number, noise_level = noise_level, noise_type = noise_type)
            cifar10_train.targets = noisy_targets.copy()
            cifar10_valid.targets = noisy_targets.copy()
        
        train_loader = DataLoader(cifar10_train, batch_size=batch_size, sampler=cifar_train_sampler)
        valid_loader = DataLoader(cifar10_valid, batch_size=batch_size, sampler=cifar_valid_sampler)
    if(dataset == "cifar100"):
        class_number = 100
        original_labels = cifar100_train.targets.copy()
        if(noise_level != 0):
            
            noisy_targets = make_noisy(cifar100_train.targets, class_number, noise_level = noise_level, noise_type = noise_type)
            cifar100_train.targets = noisy_targets.copy()
            cifar100_valid.targets = noisy_targets.copy()
        
        train_loader = DataLoader(cifar100_train, batch_size=batch_size, sampler=cifar_train_sampler)
        valid_loader = DataLoader(cifar100_valid, batch_size=batch_size, sampler=cifar_valid_sampler)
        
    if(dataset == "cub200"):
        original_labels = cub200_train.targets.copy()
        class_number = 200
        if(noise_level != 0):
            noisy_targets = make_noisy(cub200_train.targets, class_number, noise_level = noise_level, noise_type = noise_type)
            cub200_train.targets = noisy_targets
        
        train_loader = DataLoader(cub200_train, batch_size=batch_size, sampler=cub_train_sampler)
    model = ResNet34(dataset = dataset).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epoch_number/2),int(3/4*epoch_number)], gamma=0.1)
    loss_func=nn.CrossEntropyLoss()
    best_valid_accuracy = 0
    
    for epoch in range(epoch_number):
        AverageLoss=0
        model.train()
        total_samples = 0
        for i,(images,labels) in enumerate(train_loader):
            images=images.to(device)
            labels=labels.to(device)
            predictions,_ = model(images)
            loss=loss_func(predictions,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            AverageLoss+=loss.item()*images.size(0)
            total_samples += images.size(0)
        if(epoch%10 == 9 and dataset != "cub200"):
            valid_accuracy, validloss=calculate_valid_acc(model, valid_loader)
            if(valid_accuracy > best_valid_accuracy):
                best_valid_accuracy = valid_accuracy
        
        scheduler.step()
      
    if(dataset == "cifar10"):
        cifar10_train.targets = original_labels.copy()
    if(dataset == "cifar100"):
        cifar100_train.targets = original_labels.copy()
    if(dataset == "cub200"):
        cub200_train.targets = original_labels.copy()
    return best_valid_accuracy


def train_ols(dataset = "cifar10", noise_level = 0, noise_type = "symmetric",  batch_size = 128,  lr = 0.1, momentum = 0.9, weight_decay = 5e-4, epoch_number = 200):
    if(dataset == "cifar10"):
        original_labels = cifar10_train.targets.copy()
        class_number = 10
        if(noise_level != 0):
            noisy_targets = make_noisy(cifar10_train.targets, class_number, noise_level = noise_level, noise_type = noise_type)
            cifar10_train.targets = noisy_targets.copy()
            cifar10_valid.targets = noisy_targets.copy()
        
        train_loader = DataLoader(cifar10_train, batch_size=batch_size, sampler=cifar_train_sampler)
        valid_loader = DataLoader(cifar10_valid, batch_size=batch_size, sampler=cifar_valid_sampler)
    if(dataset == "cifar100"):
        class_number = 100
        original_labels = cifar100_train.targets.copy()
        if(noise_level != 0):
            
            noisy_targets = make_noisy(cifar100_train.targets, class_number, noise_level = noise_level, noise_type = noise_type)
            cifar100_train.targets = noisy_targets.copy()
            cifar100_valid.targets = noisy_targets.copy()
        
        train_loader = DataLoader(cifar100_train, batch_size=batch_size, sampler=cifar_train_sampler)
        valid_loader = DataLoader(cifar100_valid, batch_size=batch_size, sampler=cifar_valid_sampler)
        
    if(dataset == "cub200"):
        original_labels = cub200_train.targets.copy()
        class_number = 200
        if(noise_level != 0):
            noisy_targets = make_noisy(cub200_train.targets, class_number, noise_level = noise_level, noise_type = noise_type)
            cub200_train.targets = noisy_targets
        
        train_loader = DataLoader(cub200_train, batch_size=batch_size, sampler=cub_train_sampler)
    model = ResNet34(dataset = dataset).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epoch_number/2),int(3/4*epoch_number)], gamma=0.1)
    
    loss_func=nn.CrossEntropyLoss()

    best_valid_accuracy = 0
    
    soft_labels_old = torch.ones(class_number, class_number, dtype=torch.float32).to(device)*(1/class_number)
    
    soft_labels_old.require_grad = False
    
    counts_old = torch.zeros(class_number, dtype=torch.float32).to(device)
    counts_old.require_grad = False
    counts_new = torch.zeros(class_number, dtype=torch.float32).to(device)
    counts_new.require_grad = False
    for epoch in range(epoch_number):
        soft_labels_new = torch.zeros(class_number, class_number, dtype=torch.float32).to(device)
        soft_labels_new.require_grad = False
        counts_new = torch.zeros(class_number, dtype=torch.float32).to(device)
        counts_new.require_grad = False
        AverageLoss=0
        total_samples = 0
        model.train()
        for i,(images,labels) in enumerate(train_loader):
            images=images.to(device)
            labels=labels.to(device)
            predictions,_ = model(images)
           
            loss = 0.5*(loss_func(predictions,labels) + soft_cross_entropy(predictions,labels, soft_labels_old))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            with torch.no_grad():
           
                for k in range(images.size(0)):
                    logits = torch.softmax(predictions, dim = 1)
                    predicted_category = torch.argmax(predictions[k])
                    if (int(predicted_category) == int(labels[k])):
                       
                        soft_labels_new[int(labels[k])]+=logits[k]
                        counts_new[int(labels[k])] += 1
                
            AverageLoss+=loss.item()*images.size(0)
            total_samples += images.size(0)
        with torch.no_grad():
            
            for i in range(class_number): 
                if(int(counts_new[i]) == 0):
                    soft_labels_new[i] = 1/class_number
                else:
                    soft_labels_new[i] = soft_labels_new[i]/counts_new[i]
        
        soft_labels_old = (soft_labels_new).clone().detach()
        
        
        if(epoch%10 == 9 and dataset != "cub200"):
            valid_accuracy, validloss=calculate_valid_acc(model, valid_loader)
            if(valid_accuracy > best_valid_accuracy):
                best_valid_accuracy = valid_accuracy
        scheduler.step()
        
    if(dataset == "cifar10"):
        cifar10_train.targets = original_labels.copy()
    if(dataset == "cifar100"):
        cifar100_train.targets = original_labels.copy()
    if(dataset == "cub200"):
        cub200_train.targets = original_labels.copy()
    return best_valid_accuracy, soft_labels_old[3]

def train_mbls(Lambda = 0.1, m = 6, dataset = "cifar10", noise_level = 0, noise_type = "symmetric",  batch_size = 128, lr = 0.1, momentum = 0.9, weight_decay = 5e-4, epoch_number = 200):
    
   
    
    if(dataset == "cifar10"):
        original_labels = cifar10_train.targets.copy()
        class_number = 10
        if(noise_level != 0):
            noisy_targets = make_noisy(cifar10_train.targets, class_number, noise_level = noise_level, noise_type = noise_type)
            cifar10_train.targets = noisy_targets.copy()
            cifar10_valid.targets = noisy_targets.copy()
        
        train_loader = DataLoader(cifar10_train, batch_size=batch_size, sampler=cifar_train_sampler)
        valid_loader = DataLoader(cifar10_valid, batch_size=batch_size, sampler=cifar_valid_sampler)
    if(dataset == "cifar100"):
        class_number = 100
        original_labels = cifar100_train.targets.copy()
        if(noise_level != 0):
            
            noisy_targets = make_noisy(cifar100_train.targets, class_number, noise_level = noise_level, noise_type = noise_type)
            cifar100_train.targets = noisy_targets.copy()
            cifar100_valid.targets = noisy_targets.copy()
        
        train_loader = DataLoader(cifar100_train, batch_size=batch_size, sampler=cifar_train_sampler)
        valid_loader = DataLoader(cifar100_valid, batch_size=batch_size, sampler=cifar_valid_sampler)
        
    if(dataset == "cub200"):
        original_labels = cub200_train.targets.copy()
        class_number = 200
        if(noise_level != 0):
            noisy_targets = make_noisy(cub200_train.targets, class_number, noise_level = noise_level, noise_type = noise_type)
            cub200_train.targets = noisy_targets
        
        train_loader = DataLoader(cub200_train, batch_size=batch_size, sampler=cub_train_sampler)
    
    model = ResNet34(dataset = dataset).to(device)  
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epoch_number/2),int(3/4*epoch_number)], gamma=0.1)
    best_valid_accuracy = 0
    
    
    loss_func=nn.CrossEntropyLoss()
    for epoch in range(epoch_number):
        AverageLoss=0
        MarginLoss = 0
        CELoss = 0
        model.train()
        total_samples = 0
        for i,(images,labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            predictions,_ = model(images)
            celoss = loss_func(predictions,labels)
            margin_loss = margin_based_loss(predictions, m)
            loss = celoss + Lambda*margin_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            AverageLoss+=loss.item()*images.size(0)
            CELoss += celoss.item()*images.size(0)
            MarginLoss += Lambda*margin_loss.item()*images.size(0)
            total_samples += images.size(0)
        if(epoch%10 == 9 and dataset != "cub200"):
            valid_accuracy, validloss=calculate_valid_acc(model, valid_loader)
            if(valid_accuracy > best_valid_accuracy):
                best_valid_accuracy = valid_accuracy
        scheduler.step()

        
    if(dataset == "cifar10"):
        cifar10_train.targets = original_labels.copy()
    if(dataset == "cifar100"):
        cifar100_train.targets = original_labels.copy()
    if(dataset == "cub200"):
        cub200_train.targets = original_labels.copy()
    return best_valid_accuracy  



loss_values=[]


def train_pencil(alpha, beta, Lambda, epochs, dataset = "cifar10", noise_level = 0, noise_type = "symmetric", batch_size = 128, lrs = [0.35, 0.2], momentum = 0.9, weight_decay = 1e-4):
    if(dataset == "cifar10"):
        
        original_labels = cifar10_train.targets.copy()
        final_targets = original_labels.copy()
        class_number = 10
        if(noise_level != 0):
            noisy_targets = make_noisy(cifar10_train.targets, class_number, noise_level = noise_level, noise_type = noise_type)
            cifar10_train.targets = noisy_targets.copy()
            cifar10_valid.targets = noisy_targets.copy()
            final_targets = noisy_targets.copy()
        Pencil_train = Dataset_with_Indices(cifar10_train)
        Pencil_valid = Dataset_with_Indices(cifar10_valid)
        
        train_loader = DataLoader(Pencil_train, batch_size=batch_size, sampler=cifar_train_sampler)
        valid_loader = DataLoader(Pencil_valid, batch_size=batch_size, sampler=cifar_valid_sampler)
        
    if(dataset == "cifar100"):
        original_labels = cifar100_train.targets.copy()
        final_targets = original_labels.copy()
        class_number = 100
        if(noise_level != 0):
            noisy_targets = make_noisy(cifar100_train.targets, class_number, noise_level = noise_level, noise_type = noise_type)
            cifar100_train.targets = noisy_targets.copy()
            cifar100_valid.targets = noisy_targets.copy()
            final_targets = noisy_targets.copy()
        Pencil_train = Dataset_with_Indices(cifar100_train)
        Pencil_valid = Dataset_with_Indices(cifar100_valid)
        train_loader = DataLoader(Pencil_train, batch_size=batch_size, sampler=cifar_train_sampler)
        valid_loader = DataLoader(Pencil_valid, batch_size=batch_size, sampler=cifar_valid_sampler)
        
    if(dataset == "cub200"):
        original_labels = cub200_train.targets.copy()
        final_targets = original_labels.copy()
        class_number  = 200
        
        if(noise_level != 0):
            noisy_targets = make_noisy(cub200_train.targets, class_number, noise_level = noise_level, noise_type = noise_type)
            cub200_train.targets = noisy_targets.copy()
            final_targets = noisy_targets.copy()
        Pencil_train = Dataset_with_Indices(cub200_train)
        train_loader = DataLoader(Pencil_train, batch_size=batch_size, sampler=cub_train_sampler)
    
    encoded_labels = F.one_hot(torch.LongTensor(final_targets), class_number).float()
    model = ResNet34_Pencil(dataset, encoded_labels).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr = lrs[0], momentum = momentum, weight_decay = weight_decay)
   
    best_valid_accuracy = 0
    
    loss_func=nn.CrossEntropyLoss()
    for epoch in range(epochs[0]):
        AverageLoss=0
        model.train()
        total_samples = 0
        for i,(images,labels, indices) in enumerate(train_loader):
            
            images=images.to(device)
            labels=labels.to(device)
            indices = indices.to(device)
            predictions, learned_labels, log_labels = model(images, indices)
            loss=loss_func(predictions,labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            AverageLoss+=loss.item()*images.size(0)
            total_samples += images.size(0)
        if(epoch%5 == 4 and dataset != "cub200"):
            valid_accuracy, validloss=pencil_valid_acc(model, valid_loader)
            if(valid_accuracy > best_valid_accuracy):
                best_valid_accuracy = valid_accuracy

        
        
        
    
    my_list = ['learned_labels.weight']
    params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))
    
    optimizer = torch.optim.SGD([{'params': base_params},
                {'params': params, 'lr': Lambda*lrs[0], "weight_decay": weight_decay/Lambda}], lr = lrs[0], momentum = momentum, weight_decay = weight_decay)
    
    best_valid_accuracy = 0

    for epoch in range(epochs[1]):
        AverageLoss=0
        CompatibilityLoss = 0
        ClassificationLoss = 0
        EntropyLoss = 0
        model.train()
        total_samples = 0
        for i,(images,labels, indices) in enumerate(train_loader):
            images=images.to(device)
            labels=labels.to(device).view(-1)
            indices = indices.to(device)
            predictions ,learned_labels, log_labels = model(images, indices)
            
            classification_loss = kl_loss(predictions, log_labels)
            compatibility_loss = loss_func(learned_labels, labels)
            entropy_loss = entropy(predictions)
          
            loss = (1/class_number)*classification_loss + alpha*compatibility_loss + (beta/class_number)*entropy_loss
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            AverageLoss+=loss.item()*images.size(0)
            total_samples += images.size(0)
        if(epoch%1 == 0 and dataset != "cub200"):
            valid_accuracy, validloss = pencil_valid_acc(model, valid_loader)
            if(valid_accuracy > best_valid_accuracy):
                best_valid_accuracy = valid_accuracy
      
    for name, param in model.named_parameters():
         if name.startswith("learned_labels"):  
             param.requires_grad = False
    optimizer = torch.optim.SGD(model.parameters(), lr = lrs[1], momentum = momentum, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs[2]/3), 2*int(epochs[2]/3)], gamma=0.1)
    
    for epoch in range(epochs[2]):
        AverageLoss=0
        total_samples = 0
        for i,(images,labels, indices) in enumerate(train_loader):
            
            images=images.to(device)
            labels=labels.to(device)
            indices = indices.to(device)
            predictions, learned_labels, log_labels = model(images, indices)
            
            classification_loss = kl_loss(predictions, log_labels)
            loss = (1/class_number)*classification_loss
            optimizer.zero_grad()
            loss.backward()
           
            optimizer.step()
            
            
            AverageLoss+=loss.item()*images.size(0)
            total_samples += images.size(0)
        if(epoch%1 == 0 and dataset != "cub200"):
            valid_accuracy, validloss=pencil_valid_acc(model, valid_loader)
            if(valid_accuracy > best_valid_accuracy):
                best_valid_accuracy = valid_accuracy
        scheduler.step()
        
    if(dataset == "cifar10"):
        cifar10_train.targets = original_labels.copy()
    if(dataset == "cifar100"):
        cifar100_train.targets = original_labels.copy()
    if(dataset == "cub200"):
        cub200_train.targets = original_labels.copy()
    return best_valid_accuracy  

def train_esls(alpha = 0.5, dataset = "cifar10", noise_level = 0, noise_type = "symmetric", batch_size = 128, lr = 0.1, momentum = 0.9, weight_decay = 5e-4, epochs = [100,100]):
    if(dataset == "cifar10"):
        original_labels = cifar10_train.targets.copy()
        
        class_number = 10
        if(noise_level != 0):
            noisy_targets = make_noisy(cifar10_train.targets, class_number, noise_level = noise_level, noise_type = noise_type)
            cifar10_train.targets = noisy_targets
            cifar10_valid.targets = noisy_targets
            
        train_loader = DataLoader(cifar10_train, batch_size=batch_size, sampler=cifar_train_sampler)
        valid_loader = DataLoader(cifar10_valid, batch_size=batch_size, sampler=cifar_valid_sampler)
        indiced_data = Dataset_with_Indices(cifar10_train)
        indiced_loader =  DataLoader(indiced_data, batch_size=batch_size, sampler=cifar_train_sampler)
    if(dataset == "cifar100"):
        original_labels = cifar100_train.targets.copy()
        
        
        class_number = 100
        if(noise_level != 0):
            noisy_targets = make_noisy(cifar100_train.targets, class_number, noise_level = noise_level, noise_type = noise_type)
            cifar100_train.targets = noisy_targets
            cifar100_valid.targets = noisy_targets
            
        train_loader = DataLoader(cifar100_train, batch_size=batch_size, sampler=cifar_train_sampler)
        valid_loader = DataLoader(cifar100_valid, batch_size=batch_size, sampler=cifar_valid_sampler)
        indiced_data = Dataset_with_Indices(cifar100_train)
        indiced_loader =  DataLoader(indiced_data, batch_size=batch_size, sampler=cifar_train_sampler)
    if(dataset == "cub200"):
        original_labels = cub200_train.targets.copy()
        
        class_number = 200
        if(noise_level != 0):
            noisy_targets = make_noisy(cub200_train.targets, class_number, noise_level = noise_level, noise_type = noise_type)
            cub200_train.targets = noisy_targets
            
        train_loader = DataLoader(cub200_train, batch_size=batch_size, sampler=cub_train_sampler)
        indiced_data = Dataset_with_Indices(cub200_train)
        indiced_loader =  DataLoader(indiced_data, batch_size=batch_size, sampler=cub_train_sampler)
    
    
    model = ResNet34(dataset = dataset).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs[0]-10,int(epochs[0]+1/2*epochs[1]-5)], gamma=0.1)
    best_valid_accuracy = 0
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(epochs[0]):
        AverageLoss=0
        model.train()
        total_samples = 0
        for i,(images,labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            predictions, embeddings = model(images)
            loss = loss_func(predictions,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            AverageLoss+=loss.item()*images.size(0)
            total_samples += images.size(0)
        if(epoch%5 == 4 and dataset != "cub200"):
            valid_accuracy, validloss = calculate_valid_acc(model, valid_loader)
            print(epoch,valid_accuracy, validloss)
            if(valid_accuracy > best_valid_accuracy):
                best_valid_accuracy = valid_accuracy
        print(epoch, AverageLoss/total_samples)     
        scheduler.step()
    avg_embeddings = calculate_class_embeddings(model, class_number, indiced_loader, dataset = dataset)
    
    
    pairwise_cos = pairwise_cosine_similarity(avg_embeddings)
    temperature = math.log(9*class_number)/(1-torch.mean(pairwise_cos).item())

    for epoch in range(epochs[1]):
        AverageLoss=0
        model.train()
        total_samples = 0
        for i,(images,labels) in enumerate(train_loader):
            images=images.to(device)
            labels=labels.to(device)
            predictions, embeddings = model(images)
            #pca_embeddings = pca.transform(embeddings.cpu().detach().numpy())
            #pca_embeddings = torch.from_numpy(pca_embeddings).to(device)
            #cos_sims = pairwise_cosine_similarity(pca_embeddings, avg_embeddings)
            cos_sims = pairwise_cosine_similarity(embeddings, avg_embeddings)
            if(i%100 == 0):
                print(cos_sims[0])
            soft_labels =  torch.nn.functional.softmax(cos_sims*temperature, dim = 1)
 
            normal_loss=loss_func(predictions,labels)
            
            loss = alpha*manual_cross_entropy(predictions, soft_labels) + (1-alpha)*normal_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            AverageLoss+=loss.item()*images.size(0)
            total_samples += images.size(0)
       
        if(epoch%1 == 0 and dataset != "cub200"):
            valid_accuracy, validloss = calculate_valid_acc(model, valid_loader)
            if(valid_accuracy > best_valid_accuracy):
                best_valid_accuracy = valid_accuracy

        scheduler.step()
      
 
   
    if(dataset == "cifar10"):
        cifar10_train.targets = original_labels.copy()
    if(dataset == "cifar100"):
        cifar100_train.targets = original_labels.copy()
    if(dataset == "cub200"):
        cub200_train.targets = original_labels.copy()
    

      
    return best_valid_accuracy

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

cifar10_normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2616])

cifar100_normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                         std=[0.2673, 0.2564, 0.2762])

cub200_normalize = transforms.Normalize(mean=[0.4856, 0.4994, 0.4324],
                                         std=[0.2264, 0.2218, 0.2606])


cifar10_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    cifar10_normalize,
])

cifar10_transform_valid = transforms.Compose([
           transforms.ToTensor(),
           cifar10_normalize
       ])

cifar100_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    cifar100_normalize,
])

cifar100_transform_valid = transforms.Compose([
           transforms.ToTensor(),
           cifar100_normalize
       ])

cub200_transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        cub200_normalize
    ])

cub200_transform_valid = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        cub200_normalize
    ])

cifar10_train = datasets.CIFAR10(root="./data",train=True,download=True,transform=cifar10_transform_train)
cifar10_valid = datasets.CIFAR10(root="./data",train=True,download=True,transform=cifar10_transform_valid)

cifar100_train = datasets.CIFAR100(root="./data",train=True,download=True,transform=cifar100_transform_train)
cifar100_valid = datasets.CIFAR100(root="./data",train=True,download=True,transform=cifar100_transform_valid)

img_path = 'data/CUB_200_2011/images/'
img_txt = 'data/CUB_200_2011/train_list.txt'

cub200_train = Cub200Dataset(img_path, img_txt, cub200_transform_train)
cub200_valid = Cub200Dataset(img_path, img_txt, cub200_transform_valid)



valid_size = 0.1


num_train = len(cifar10_train)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
cifar_train_sampler = SubsetRandomSampler(train_idx)
cifar_valid_sampler = SubsetRandomSampler(valid_idx)

num_train_cub = len(cub200_train)
cub_indices = list(range(num_train_cub))
cub_train_idx = cub_indices[:]
cub_train_sampler = SubsetRandomSampler(cub_train_idx)





#train_default(epoch_number = 100, dataset = "cub200", noise_level = 0, lr = 0.01, batch_size = 16)
#train_ols(epoch_number = 100, dataset = "cub200", lr = 0.01, batch_size = 16)
#train_mbls(m = 10, epoch_number = 100, dataset= "cub200", batch_size = 16, lr = 0.01)
#train_esls(alpha = 0.5, dataset =  "cub200", epochs = [50,50], batch_size = 16, lr = 0.01)
#train_pencil( 0.1, 0.4, 10000, [45, 80, 75], lrs = [0.35, 0.2], noise_level = 0.3, noise_type = "asymmetric", dataset = "cifar100", weight_decay = 1e-4)
#train_pencil( 0.1, 0.8, 100, [45, 80, 75], lrs = [0.01, 0.2], dataset = "cifar10", weight_decay = 1e-4)
#train_pencil(0, 0.8, 3000, [24, 40, 36], lrs = [0.002, 0.001], dataset = "cub200", weight_decay = 1e-4, batch_size = 16)
