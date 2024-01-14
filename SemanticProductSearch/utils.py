# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:13:37 2024

@author: USER
"""


import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hinge_loss (batch_of_queries, batch_of_products, positive_threshold = 0.9, negative_threshold = 0.2):
    batch_size = batch_of_queries.size(0)
    #transposed_products = torch.transpose(batch_of_products, 0, 1)
    dot_products = pairwise_cosine_similarity(batch_of_queries, batch_of_products)
    losses = dot_products - (torch.ones((batch_size, batch_size)).to(device)*(negative_threshold))+torch.eye(batch_size).to(device)*(negative_threshold - positive_threshold)
    positive_matches = torch.diag(losses)
    negative_matches = losses - torch.diag(positive_matches)
    minimum_zeroes = torch.zeros_like(positive_matches).to(device)
    maximum_zeroes = torch.zeros_like(negative_matches).to(device)
    positive_losses = torch.square(torch.minimum(positive_matches, minimum_zeroes))
    negative_losses = torch.square(torch.maximum(negative_matches, maximum_zeroes))
    loss = torch.sum(positive_losses)/(batch_size) + torch.sum(negative_losses)/((batch_size-1)*batch_size) 
    
    return loss 

def calculate_valid_loss(model, valid_loader):
    
    with torch.no_grad():
        ValidLoss = 0
        model.eval()
        total = 0
        loss_func=nn.MSELoss()
        for i,(queries, products, relevances) in enumerate(valid_loader):
            queries = queries
            products = products
            relevances = relevances.to(device).float()
            
            predictions = model(queries, products)
            
            loss=loss_func(predictions,relevances)
            
            ValidLoss+= loss.item()*len(queries)
            total += len(queries)
        
        return  ValidLoss/total    

def calculate_valid_loss_custom_tokenizer(model, valid_loader):
    
    with torch.no_grad():
        ValidLoss = 0
        model.eval()
        total = 0
        loss_func=nn.MSELoss()
        for i,(numerated_queries, numerated_products, query_lengths, product_lengths, relevances) in enumerate(valid_loader):
            numerated_queries = numerated_queries.to(device)
            numerated_products = numerated_products.to(device)
            query_lengths = query_lengths.to(device)
            product_lengths = product_lengths.to(device)
            relevances = relevances.to(device).float()
            
            predictions = model(numerated_queries, numerated_products, query_lengths, product_lengths)
            
            loss=loss_func(predictions,relevances)
            
            ValidLoss+= loss.item()*numerated_queries.size(0)
            total += numerated_queries.size(0)
        
        return  ValidLoss/total  
    
class Dataset_2(Dataset):
 
  def __init__(self, df):
    self.dataframe = df
 
  def __len__(self):
    return len(self.dataframe)
   
  def __getitem__(self,idx):
    return self.dataframe.iloc[idx,4], self.dataframe.iloc[idx,3], self.dataframe.iloc[idx,2]

class Dataset_custom_tokenizer(Dataset):
    def __init__(self, df):
      self.dataframe = df
   
    def __len__(self):
      return len(self.dataframe)
     
    def __getitem__(self,idx):
      return self.dataframe.iloc[idx,8], self.dataframe.iloc[idx,7], self.dataframe.iloc[idx,10], self.dataframe.iloc[idx,9], self.dataframe.iloc[idx,2]