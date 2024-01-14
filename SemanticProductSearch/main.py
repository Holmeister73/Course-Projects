# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
import pickle
from tokenization import numerated_df
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import  SPS_finetune, SPS_pretrain, SPS_static_embeddings, SPS_static_embeddings_with_custom_tokenizer
from utils import hinge_loss, calculate_valid_loss, calculate_valid_loss_custom_tokenizer, Dataset_2, Dataset_custom_tokenizer
from transformers import AutoTokenizer, AutoModel, AdamW, get_cosine_schedule_with_warmup



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#torch.set_float32_matmul_precision("medium")




df_train = pd.read_csv("cleaned_train.csv", encoding= "ISO-8859-1")
df_val = pd.read_csv("cleaned_validation.csv", encoding= "ISO-8859-1")
df_test = pd.read_csv("cleaned_test.csv", encoding= "ISO-8859-1")

df_only_positives = df_train[df_train["relevance"] >= 3]

train_dataset = Dataset_2(df_train)
valid_dataset = Dataset_2(df_val)
pretrain_dataset = Dataset_2(df_only_positives)


feature_based_config = {
    'model_from_huggingface': True,
    'model_name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
    'batch_size': 128,
    'embedding_dim': 384 ,
    'lr': 0.0003,
    'weight_decay': 0.001,
    'n_epochs': 5,
    'abs_diff': True,
    'cos_sim': True
}

finetune_config = {
    'model_from_huggingface': False,
    'pickle_model': 'pretrained_10_epochs.pk',
    'model_name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
    'unfreeze_starting_with_this_layer': 5,
    'batch_size': 64,
    'embedding_dim': 384,
    'lr': 0.0001,
    'weight_decay': 0.001,
    'n_epochs': 5,
    'abs_diff': True,
    'cos_sim': True
}
                   

pretrain_config = {
    'model_name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
    'unfreeze_starting_with_this_layer': 5,
    'batch_size': 64,
    'embedding_dim': 384,
    'lr': 0.0001,
    'weight_decay': 0.001,
    'n_epochs': 10,
    'abs_diff': True,
    'cos_sim': True
}


static_emb_config = {
    'model_name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
    'batch_size': 128,
    'embedding_dim': 384,
    'lr': 0.005,
    'weight_decay': 0.001,
    'n_epochs': 10,
    'abs_diff': True,
    'cos_sim': True
    
}

static_emb_with_custom_tokenizer_config = {
    'batch_size': 128,
    'embedding_dim': 64,
    'vocab_size': 45001,
    'lr': 0.0003,
    'weight_decay': 0.001,
    'n_epochs': 6,
    'abs_diff': True,
    'cos_sim': True
}




    

    
  

def feature_based(config):
    model = SPS_finetune(config).to(device)  #finetune model can be used for feature based just by freezing the pretrained model's layers
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    loss_func = nn.MSELoss()
    for name, param in model.named_parameters():
        if name.startswith("dense") == False:
            param.requires_grad = False
             
    
    train_loader = DataLoader(train_dataset, batch_size = finetune_config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size = finetune_config["batch_size"], shuffle=False)
    for epoch in range(config["n_epochs"]):
        AverageLoss=0
        model.train()
        total_samples = 0
        for i,(queries, products, relevances) in enumerate(train_loader):
            relevances = relevances.to(device).float()
            predictions = model(queries, products)
            loss=loss_func(predictions, relevances)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            AverageLoss+=loss.item()*len(queries)
            total_samples += len(queries)
            
        if(epoch%1 == 0):
            validloss=calculate_valid_loss(model, valid_loader)
            print(validloss)
        print(epoch, AverageLoss/total_samples)
        
        
    with open("minilm_5_feature_based_no_cos_no_abs.pk", "wb") as filename:
        pickle.dump(model, filename)
    return 

def finetune(config):
    model = SPS_finetune(config).to(device) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    loss_func = nn.MSELoss()      
    train_loader = DataLoader(train_dataset, batch_size = finetune_config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size = finetune_config["batch_size"], shuffle=False)
    if config["model_from_huggingface"] == True:
        for name, param in model.named_parameters():
            if name.startswith("dense") == False:
                param.requires_grad = False
        if 0 <= config["unfreeze_starting_with_this_layer"] <= 5:
            for name, param in model.named_parameters():
                for i in range(config["unfreeze_starting_with_this_layer"],6):
                    if name.startswith("pretrained_model.encoder.layer."+str(i)):  
                        param.requires_grad = True
                if name.startswith("pretrained_model.pooler"):
                    param.requires_grad = True
    else:
        
        for name, param in model.named_parameters():
            if name.startswith("dense") == False:
                param.requires_grad = False
        if 0 <= config["unfreeze_starting_with_this_layer"] <= 5:
            for name, param in model.named_parameters():
                for i in range(config["unfreeze_starting_with_this_layer"],6):
                    if name.startswith("pretrained_model.pretrained_model.encoder.layer."+str(i)):  
                        param.requires_grad = True
                if name.startswith("pretrained_model.pretrained_model.pooler"):
                    param.requires_grad = True
    for name, param in model.named_parameters():
        print(name, param.requires_grad)          
    for epoch in range(config["n_epochs"]):
        AverageLoss=0
        model.train()
        total_samples = 0
        for i,(queries, products, relevances) in enumerate(train_loader):
            relevances = relevances.to(device).float()
            predictions = model(queries, products)
            loss=loss_func(predictions, relevances)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            AverageLoss+=loss.item()*len(queries)
            total_samples += len(queries)
            
        if(epoch%5 == 1):
            validloss=calculate_valid_loss(model, valid_loader)
            print(validloss)
        print(epoch, AverageLoss/total_samples)
        
        
    with open("minilm_5_finetune_with_custom_pretrain.pk", "wb") as filename:
        pickle.dump(model, filename)
    return 

def pretrain(config):
    model = SPS_pretrain(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(config["n_epochs"]/2), int(config["n_epochs"]*3/4)], gamma=0.5)
    train_loader = DataLoader(pretrain_dataset, batch_size = pretrain_config["batch_size"], shuffle=True)
    for name, param in model.named_parameters():
        if name.startswith("dense") == False:
            param.requires_grad = False
    if 0 <= config["unfreeze_starting_with_this_layer"] <= 5:
        for name, param in model.named_parameters():
            for i in range(config["unfreeze_starting_with_this_layer"],6):
                if name.startswith("pretrained_model.encoder.layer."+str(i)):  
                    param.requires_grad = True
            if name.startswith("pretrained_model.pooler"):
                param.requires_grad = True
    for epoch in range(config["n_epochs"]):
        
        AverageLoss=0
        model.train()
        total_samples = 0
        for i,(queries, products, relevances) in enumerate(train_loader):
            relevances = relevances.to(device).float()
            query_embs, product_embs = model(queries, products)
            loss = hinge_loss(query_embs, product_embs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            AverageLoss+=loss.item()*len(queries)
            total_samples += len(queries)
            
        print(epoch, AverageLoss/total_samples)
        scheduler.step()
        
    with open("pretrained_10_epochs.pk", "wb") as filename:
        pickle.dump(model, filename)
        
    return

def static_embedding(config):
    model = SPS_static_embeddings(config).to(device) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    loss_func = nn.MSELoss()      
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 2)
    train_dataset = Dataset_2(df_train)
    train_loader = DataLoader(train_dataset, batch_size = finetune_config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size = finetune_config["batch_size"], shuffle=False)
    
    for epoch in range(config["n_epochs"]):
        AverageLoss=0
        model.train()
        total_samples = 0
        for i,(queries, products, relevances) in enumerate(train_loader):
            relevances = relevances.to(device).float()
            predictions = model(queries, products)
            loss=loss_func(predictions, relevances)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            AverageLoss+=loss.item()*len(queries)
            total_samples += len(queries)
            
        if(epoch%1 == 0):
            validloss=calculate_valid_loss(model, valid_loader)
            print(validloss)
        print(epoch, AverageLoss/total_samples)
        scheduler.step(validloss)
    with open("static_embedding_10.pk", "wb") as filename:
        pickle.dump(model, filename)
    return 

def static_embedding_with_custom_tokenizer(config):
    model = SPS_static_embeddings_with_custom_tokenizer(config).to(device) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    loss_func = nn.MSELoss()      
    df_train, df_valid, df_test = numerated_df()  # This function returns the custom tokenized version of the datasets. It contains token ids.
    train_dataset = Dataset_custom_tokenizer(df_train)
    valid_dataset = Dataset_custom_tokenizer(df_valid)
    train_loader = DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size = config["batch_size"], shuffle = False)
    for epoch in range(config["n_epochs"]):
        AverageLoss=0
        model.train()
        total_samples = 0
        for i,(numerated_queries, numerated_products, query_lengths, product_lengths, relevances) in enumerate(train_loader):
            numerated_queries = numerated_queries.to(device)
            numerated_products = numerated_products.to(device)
            query_lengths = query_lengths.to(device)
            product_lengths = product_lengths.to(device)
            relevances = relevances.to(device).float()
            
            predictions = model(numerated_queries, numerated_products, query_lengths, product_lengths)
            
            loss=loss_func(predictions,relevances)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            AverageLoss+= loss.item()*numerated_queries.size(0)
            total_samples += numerated_queries.size(0)
        if(epoch%1 == 0):
            validloss=calculate_valid_loss_custom_tokenizer(model, valid_loader)
            print(validloss)
            
        print(epoch, AverageLoss/total_samples)
        
    with open("static_embedding_custom_tokenizer_6.pk", "wb") as filename:
        pickle.dump(model, filename)
    
    return
    
    
print(len(df_train))
print(len(df_only_positives))

#finetune(finetune_config)
#feature_based(feature_based_config)
#pretrain(pretrain_config)
#static_embedding(static_emb_config)
static_embedding_with_custom_tokenizer(static_emb_with_custom_tokenizer_config)


