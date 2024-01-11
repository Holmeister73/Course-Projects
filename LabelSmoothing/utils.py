# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 21:39:22 2024

@author: USER
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

def one_hot_encode(label_list, num_classes):
    
    encoded = torch.FloatTensor([[0 for i in range(num_classes)] for j in range(len(label_list))])
    for i in range(len(label_list)):
        encoded[i, label_list[i]] = 1

    return encoded

def margin_based_loss(output, m):
    
    max_values = output.max(dim=1)
    max_values = max_values.values.unsqueeze(dim=1).repeat(1, output.shape[1])
    diff = max_values - output
    diff_with_margin = diff - m
    loss = F.relu(diff_with_margin).mean()

    return  loss
    
def soft_cross_entropy(output, target, soft_labels):    
    
    target_prob = torch.zeros_like(output)
    batch = output.shape[0]
    for k in range(batch):
            target_prob[k] = soft_labels[int(target[k])]
            
    log_like = -torch.nn.functional.log_softmax(output, dim=1)
    loss = torch.sum(torch.mul(log_like, target_prob)) / batch 
    return loss

def cross_entropy_without_softmax(predictions, labels, num_classes):
    
    batch = predictions.shape[0]
    log_predictions = -torch.log(predictions)
    labels = F.one_hot(labels, num_classes).float()
    loss = torch.sum(torch.mul(log_predictions,labels)) / batch
    
    return loss

def manual_cross_entropy(predictions, labels):
    batch = predictions.shape[0]
    log_softmax_predictions = -torch.nn.functional.log_softmax(predictions, dim=1)
    loss = torch.sum(torch.mul(log_softmax_predictions,labels)) / batch
    
    return loss

def entropy(prediction):
    batch = prediction.shape[0]
    prediction_log_softmax = -torch.nn.functional.log_softmax(prediction, dim=1)
    prediction_softmax = torch.nn.functional.softmax(prediction, dim = 1)
    loss = torch.sum(torch.mul(prediction_log_softmax, prediction_softmax)) / batch
    
    return loss

def kl_loss(predictions, log_labels):
    batch = predictions.shape[0]
    predictions_softmax = torch.nn.functional.softmax(predictions, dim = 1)
    log_softmax_predictions = torch.nn.functional.log_softmax(predictions, dim=1)
    loss = torch.sum(torch.mul(predictions_softmax,(log_softmax_predictions-log_labels))) / batch
    
    return loss
def calculate_valid_acc(network, valid_loader):
    
    with torch.no_grad():
        ValidLoss = 0
        network.eval()
        samples=0
        correct=0
        total = 0
        loss_func=nn.CrossEntropyLoss()
        for j,(images,labels) in enumerate(valid_loader):
            images=images.to(device)
            labels=labels.to(device)
            
            predictions,_ = network(images)
            
            loss=loss_func(predictions,labels)
            
            _,predicted=torch.max(predictions,1)
            samples+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            ValidLoss+= loss.item()*labels.size(0)
            total += labels.size(0)
    
    acc=correct/samples
    
    return acc, ValidLoss/total      


def pencil_valid_acc(network, valid_loader):
    with torch.no_grad():
        ValidLoss = 0
        network.eval()
        samples=0
        correct=0
        total = 0
        loss_func=nn.CrossEntropyLoss()
        for j,(images,labels, indices) in enumerate(valid_loader):
            images=images.to(device)
            labels=labels.to(device)
            indices = indices.to(device)
            
            predictions, learned_labels, log_labels = network(images, indices)
            
            loss=loss_func(predictions,labels)
            
            _,predicted=torch.max(predictions,1)
            
            samples+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            ValidLoss+= loss.item()*labels.size(0)
            total += labels.size(0)
    acc=correct/samples
    return acc, ValidLoss/total  

def symmetric_new_label(label, class_number):
    
    rng = np.random.default_rng()
    x = [i for i in range(label)] + [i for i in range(label+1,class_number)]
    x = np.array(x)
    
    noisy_label = int(rng.choice(x, size= 1))
    
    return noisy_label

def asymmetric_new_label(label, class_number):
    
    return (label+1)%class_number
    
def make_noisy(targets, class_number, noise_level = 0, noise_type = "symmetric"):
    
    rng = np.random.default_rng()
    class_number = class_number
    indices_to_change = list((rng.choice(np.arange(int(len(targets)*0.1),len(targets)), size = int(np.floor(noise_level*len(targets)*0.9)))))
    new_targets = targets.copy()
    for index in indices_to_change:
        if(noise_type == "symmetric"):
            new_targets[index] = symmetric_new_label(new_targets[index], class_number)        
        if(noise_type == "asymmetric"):
            new_targets[index] = asymmetric_new_label(new_targets[index], class_number)   
    
    
    
    return new_targets

def calculate_class_embeddings(model, class_number, train_loader, embedding_dim = 512, dataset = "cifar10", method = "default"):
    
    total_embeddings = torch.zeros((class_number, 512)).to(device)
    class_counts = torch.zeros(class_number).to(device)
    all_embeddings = []
    avg_embeddings = [0 for i in range(class_number)]
    Label_index=[[] for i in range(class_number)]
    with torch.no_grad():
        model.eval()
        
        for i,(images,labels, indices) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            indices = indices.to(device)
            predictions, embeddings = model(images)
            
            for j in range(labels.size(0)):
                if(method == "default"):
                    total_embeddings[int(labels[j])] += embeddings[j]
                    class_counts[int(labels[j])] += 1
                if(method == "pca"):
                    all_embeddings.append(embeddings[j].cpu().detach().numpy())
                    if(dataset.startswith("cifar")):
                        Label_index[int(labels[j])].append(int(indices[j])-5000)
                    else:
                        Label_index[int(labels[j])].append(int(indices[j]))
        
        if(method == "default"):
            class_counts = class_counts.view(class_counts.size(0), -1)
            avg_embeddings = total_embeddings/class_counts
            return avg_embeddings
    
    all_embeddings = np.array(all_embeddings)
    if(method == "pca"):
        pca = PCA(n_components = embedding_dim)
        dim_reduced_data = pca.fit_transform(all_embeddings)
        
    for k in range(class_number):
        label_indices = Label_index[k]
        #print(len(label_indices))
        label_k = np.take(dim_reduced_data, label_indices, axis = 0)
        
        avg_embeddings[k] = np.sum(label_k, axis = 0)/(label_k.shape[0])
    
    avg_embeddings = np.array(avg_embeddings)
    avg_embeddings = torch.FloatTensor(avg_embeddings).to(device)
    return avg_embeddings, pca