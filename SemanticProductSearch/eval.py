# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:29:06 2023

@author: USER
"""

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
from utils import Dataset_2, Dataset_custom_tokenizer
from transformers import AutoTokenizer,AutoModel
import gc
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df_test = pd.read_csv("cleaned_test.csv", encoding= "ISO-8859-1")

test_dataset = Dataset_2(df_test)

test_loader = DataLoader(test_dataset, batch_size = 128, shuffle=False)

with open("static_embedding_custom_tokenizer_6.pk", "rb") as filename:
    model = pickle.load(filename)

def calculate_rmse(model):
    with torch.no_grad():
        mseLoss = 0
        model.eval()
        total = 0
        loss_func=nn.MSELoss()
        for i,(queries, products, relevances) in enumerate(test_loader):
            queries = queries
            products = products
            relevances = relevances.to(device).float()
            
            predictions = model(queries, products)
            
            loss=loss_func(predictions,relevances)
            
            mseLoss+= loss.item()*len(queries)
            total += len(queries)
        
        return  math.sqrt(mseLoss/total)    
    
def calculate_rmse_custom_tokenizer(model):
    df_train, df_valid, df_test = numerated_df()
    test_dataset  = Dataset_custom_tokenizer(df_test)
    test_loader = DataLoader(test_dataset, batch_size = 128, shuffle = False)
    with torch.no_grad():
        mseLoss = 0
        model.eval()
        total = 0
        loss_func=nn.MSELoss()
        for i,(numerated_queries, numerated_products, query_lengths, product_lengths, relevances) in enumerate(test_loader):
            numerated_queries = numerated_queries.to(device)
            numerated_products = numerated_products.to(device)
            query_lengths = query_lengths.to(device)
            product_lengths = product_lengths.to(device)
            relevances = relevances.to(device).float()
            
            predictions = model(numerated_queries, numerated_products, query_lengths, product_lengths)
            
            loss=loss_func(predictions,relevances)
            
            mseLoss+= loss.item()*numerated_queries.size(0)
            total += numerated_queries.size(0)
        
        return  math.sqrt(mseLoss/total)    
    
#print(calculate_rmse(model))
print(calculate_rmse_custom_tokenizer(model))



