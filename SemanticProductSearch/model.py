# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 18:09:41 2024

@author: USER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
import pytorch_lightning as pl
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_pooling(model_output, attention_mask): # code from https://huggingface.co/transformers/v4.8.2/training.html
    
    token_embeddings = model_output.last_hidden_state
    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode(texts, tokenizer, model, max_token_len): # code from https://huggingface.co/transformers/v4.8.2/training.html
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
    
    model_output = model(**encoded_input, return_dict=True)

    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings

class SPS_finetune(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if(config['model_from_huggingface'] == True): 
            self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
            self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict = True)
        else:
            with open(config["pickle_model"], "rb") as filename:
                self.pretrained_model = pickle.load(filename)
            self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        
        if(config["abs_diff"] == True and config["cos_sim"] == True):
            if(config["embedding_dim"] == 384):
                self.dense1 = nn.Linear(config["embedding_dim"]*3, 192)
                self.dense2 = nn.Linear(192,96)
                self.dense3 = nn.Linear(97,1)
            else:
                self.dense1 = nn.Linear(config["embedding_dim"]*3, 96)
                self.dense2 = nn.Linear(96,64)
                self.dense3 = nn.Linear(65,1)
                
        if(config["abs_diff"] == True and config["cos_sim"] == False):
            if(config["embedding_dim"] == 384):
                self.dense1 = nn.Linear(config["embedding_dim"]*3, 192)
                self.dense2 = nn.Linear(192,96)
                self.dense3 = nn.Linear(96,1)
            else:
                self.dense1 = nn.Linear(config["embedding_dim"]*3, 96)
                self.dense2 = nn.Linear(96,64)
                self.dense3 = nn.Linear(64,1)
        if(config["abs_diff"] == False and config["cos_sim"] == True):
            if(config["embedding_dim"] == 384):
                self.dense1 = nn.Linear(config["embedding_dim"]*2, 288)
                self.dense2 = nn.Linear(288,64)
                self.dense3 = nn.Linear(65,1)
            else:
                self.dense1 = nn.Linear(config["embedding_dim"]*2, 144)
                self.dense2 = nn.Linear(144,64)
                self.dense3 = nn.Linear(65,1)
                
        if(config["abs_diff"] == False and config["cos_sim"] == False):
            if(config["embedding_dim"] == 384):
                self.dense1 = nn.Linear(config["embedding_dim"]*2, 288)
                self.dense2 = nn.Linear(288,64)
                self.dense3 = nn.Linear(64,1)
            else:
                self.dense1 = nn.Linear(config["embedding_dim"]*2, 144)
                self.dense2 = nn.Linear(144,64)
                self.dense3 = nn.Linear(64,1)
            
        self.tanh = nn.Tanh()
        self.cos = nn.CosineSimilarity(dim=1)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, queries, products):
        if(self.config["model_from_huggingface"] == True):
            query_avg_emb = encode(list(queries), self.tokenizer, self.pretrained_model, 8).to(device)
            product_avg_emb = encode(list(products), self.tokenizer, self.pretrained_model, 8).to(device)
        
        else:
            query_avg_emb, product_avg_emb = self.pretrained_model(queries,products)
        cos_sim = self.cos(query_avg_emb, product_avg_emb)
        cos_sim = cos_sim.unsqueeze(1)
        abs_diff = torch.abs(torch.sub(query_avg_emb,product_avg_emb))
        if(self.config["abs_diff"] == True and self.config["cos_sim"] == True):
            dense1_input = torch.cat((query_avg_emb,product_avg_emb,abs_diff),1)
            dense1_output = self.dropout(self.relu(self.dense1(dense1_input)))
            dense2_output = self.relu(self.dense2(dense1_output))
            dense3_input = torch.cat((dense2_output,cos_sim), 1)
            
        if(self.config["abs_diff"] == True and self.config["cos_sim"] == False):
            dense1_input = torch.cat((query_avg_emb,product_avg_emb,abs_diff),1)
            dense1_output = self.dropout(self.relu(self.dense1(dense1_input)))
            dense2_output = self.relu(self.dense2(dense1_output))
            dense3_input = dense2_output
            
        if(self.config["abs_diff"] == False and self.config["cos_sim"] == True):
            dense1_input = torch.cat((query_avg_emb,product_avg_emb),1)
            dense1_output = self.dropout(self.relu(self.dense1(dense1_input)))
            dense2_output = self.relu(self.dense2(dense1_output))
            dense3_input = torch.cat((dense2_output,cos_sim), 1)
            
        if(self.config["abs_diff"] == False and self.config["cos_sim"] == False):
            dense1_input = torch.cat((query_avg_emb,product_avg_emb),1)
            dense1_output = self.dropout(self.relu(self.dense1(dense1_input)))
            dense2_output = self.relu(self.dense2(dense1_output))
            dense3_input = dense2_output
            
        prediction = self.dense3(dense3_input)
        
        prediction = torch.reshape(prediction,(-1,))
        return prediction

class SPS_pretrain(nn.Module):

  def __init__(self, config: dict):
    super().__init__()
    self.config = config
    
    self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict = True)
                
  def forward(self, queries, products):
    query_avg_emb = encode(list(queries), self.tokenizer, self.pretrained_model, 8).to(device)
    product_avg_emb = encode(list(products), self.tokenizer, self.pretrained_model, 8).to(device)

    return  query_avg_emb, product_avg_emb
    
          
class SPS_static_embeddings(nn.Module):

  def __init__(self, config: dict):
    super().__init__()
    self.config = config
    self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    self.embedding = nn.Embedding(self.tokenizer.vocab_size, self.config["embedding_dim"], padding_idx = self.tokenizer.pad_token_id)
    
    if(config["abs_diff"] == True and config["cos_sim"] == True):
        self.dense1 = nn.Linear(config["embedding_dim"]*3, 288)
        self.dense2 = nn.Linear(288,96)
        self.dense3 = nn.Linear(97,1)
    if(config["abs_diff"] == True and config["cos_sim"] == False):
        self.dense1 = nn.Linear(config["embedding_dim"]*3, 288)
        self.dense2 = nn.Linear(288,96)
        self.dense3 = nn.Linear(96,1)
    if(config["abs_diff"] == False and config["cos_sim"] == True):
        self.dense1 = nn.Linear(config["embedding_dim"]*2, 432)
        self.dense2 = nn.Linear(432,64)
        self.dense3 = nn.Linear(65,1)
    if(config["abs_diff"] == False and config["cos_sim"] == False):
        self.dense1 = nn.Linear(config["embedding_dim"]*2, 432)
        self.dense2 = nn.Linear(432,64)
        self.dense3 = nn.Linear(64,1)
        
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()
    self.cos = nn.CosineSimilarity(dim=1)
    self.batchnorm = nn.BatchNorm1d(self.config["embedding_dim"])
    self.dropout = nn.Dropout(0.4)
  def forward(self, queries, products):

    query_batch = self.tokenizer(list(queries), padding = True, truncation = True, max_length = 8, return_tensors = "pt").to(device)
    product_batch = self.tokenizer(list(products), padding = True, truncation = True, max_length = 8, return_tensors = "pt").to(device)
    query_embs = self.dropout(self.embedding(query_batch.input_ids))
    product_embs = self.dropout(self.embedding(product_batch.input_ids))
    query_embs = torch.sum(query_embs, dim = 1)
    product_embs = torch.sum(product_embs, dim = 1)
    query_lengths = [[] for i in range(len(list(queries)))]
    product_lengths = [[] for i in range(len(list(products)))]
    for i in range(len(list(queries))):
        j = 0
        for input_id in list(query_batch.input_ids[i]):
         
            
            if(input_id.item() == self.tokenizer.pad_token_id):
                break
            j += 1
        query_lengths[i].append(j)
        j = 0
        for input_id in list(product_batch.input_ids[i]):
            
            
            if(input_id.item() == self.tokenizer.pad_token_id):
                break
            j += 1
        product_lengths[i].append(j)
        
        
    query_lengths = torch.FloatTensor(query_lengths).to(device)
    product_lengths = torch.FloatTensor(product_lengths).to(device)
    query_lengths = query_lengths.view((-1,1))
    product_lengths = product_lengths.view((-1,1))
    query_avg_emb = torch.div(query_embs, query_lengths)
    product_avg_emb = torch.div(product_embs, product_lengths)
    
    cos_sim = self.cos(query_avg_emb, product_avg_emb)    
    cos_sim = cos_sim.unsqueeze(1)
    abs_diff = torch.abs(torch.sub(query_avg_emb,product_avg_emb))
    if(self.config["abs_diff"] == True and self.config["cos_sim"] == True):
        dense1_input = torch.cat((query_avg_emb,product_avg_emb,abs_diff),1)
        dense1_output = self.dropout(self.relu(self.dense1(dense1_input)))
        dense2_output = self.relu(self.dense2(dense1_output))
        dense3_input = torch.cat((dense2_output,cos_sim), 1)
        
    if(self.config["abs_diff"] == True and self.config["cos_sim"] == False):
        dense1_input = torch.cat((query_avg_emb,product_avg_emb,abs_diff),1)
        dense1_output = self.dropout(self.relu(self.dense1(dense1_input)))
        dense2_output = self.relu(self.dense2(dense1_output))
        dense3_input = dense2_output
        
    if(self.config["abs_diff"] == False and self.config["cos_sim"] == True):
        dense1_input = torch.cat((query_avg_emb,product_avg_emb),1)
        dense1_output = self.dropout(self.relu(self.dense1(dense1_input)))
        dense2_output = self.relu(self.dense2(dense1_output))
        dense3_input = torch.cat((dense2_output,cos_sim), 1)
        
    if(self.config["abs_diff"] == False and self.config["cos_sim"] == False):
        dense1_input = torch.cat((query_avg_emb,product_avg_emb),1)
        dense1_output = self.dropout(self.relu(self.dense1(dense1_input)))
        dense2_output = self.relu(self.dense2(dense1_output))
        dense3_input = dense2_output
        
    prediction = self.dense3(dense3_input)
    prediction = torch.reshape(prediction,(-1,))
    
    return   prediction

class SPS_static_embeddings_with_custom_tokenizer(nn.Module):

  def __init__(self, config: dict):
    super().__init__()
    self.config = config
    self.embedding = nn.Embedding(self.config["vocab_size"], self.config["embedding_dim"], padding_idx = 0)
    
    if(config["abs_diff"] == True and config["cos_sim"] == True):
        self.dense1 = nn.Linear(config["embedding_dim"]*3, 288)
        self.dense2 = nn.Linear(288,96)
        self.dense3 = nn.Linear(97,1)
    if(config["abs_diff"] == True and config["cos_sim"] == False):
        self.dense1 = nn.Linear(config["embedding_dim"]*3, 288)
        self.dense2 = nn.Linear(288,96)
        self.dense3 = nn.Linear(96,1)
    if(config["abs_diff"] == False and config["cos_sim"] == True):
        self.dense1 = nn.Linear(config["embedding_dim"]*2, 432)
        self.dense2 = nn.Linear(432,64)
        self.dense3 = nn.Linear(65,1)
    if(config["abs_diff"] == False and config["cos_sim"] == False):
        self.dense1 = nn.Linear(config["embedding_dim"]*2, 432)
        self.dense2 = nn.Linear(432,64)
        self.dense3 = nn.Linear(64,1)
        
    self.tanh = nn.Tanh()
    self.cos = nn.CosineSimilarity(dim=1)
    self.dropout = nn.Dropout(0.6)
    self.batchnorm = nn.BatchNorm1d(self.config["embedding_dim"])
    self.relu = nn.ReLU()
    
  def forward(self, numerated_queries, numerated_products, query_lengths, product_lengths):
    
    numerated_queries = numerated_queries[:, :8]
    numerated_products = numerated_products[:, :8]
    query_embs = self.dropout(self.embedding(numerated_queries))
    product_embs = self.dropout(self.embedding(numerated_products))
    eight = torch.FloatTensor([8]).to(device)
    query_lengths = torch.minimum(query_lengths,eight).to(torch.int32)
    product_lengths = torch.minimum(product_lengths,eight).to(torch.int32)
    query_lengths = query_lengths.view((-1,1))
    product_lengths = product_lengths.view((-1,1))
    query_embs = torch.sum(query_embs, dim = 1)
    product_embs = torch.sum(product_embs, dim = 1)
    query_avg_emb = torch.div(query_embs, query_lengths)
    product_avg_emb = torch.div(product_embs, product_lengths)
    query_avg_emb = F.normalize(query_avg_emb, p = 2, dim = 1)
    product_avg_emb = F.normalize(product_avg_emb, p = 2, dim = 1)
    
    
    cos_sim = self.cos(query_avg_emb, product_avg_emb)    
    cos_sim = cos_sim.unsqueeze(1)
    abs_diff = torch.abs(torch.sub(query_avg_emb,product_avg_emb))
    if(self.config["abs_diff"] == True and self.config["cos_sim"] == True):
        dense1_input = torch.cat((query_avg_emb,product_avg_emb,abs_diff),1)
        dense1_output = self.dropout(self.relu(self.dense1(dense1_input)))
        dense2_output = self.relu(self.dense2(dense1_output))
        dense3_input = torch.cat((dense2_output,cos_sim), 1)
        
    if(self.config["abs_diff"] == True and self.config["cos_sim"] == False):
        dense1_input = torch.cat((query_avg_emb,product_avg_emb,abs_diff),1)
        dense1_output = self.dropout(self.relu(self.dense1(dense1_input)))
        dense2_output = self.relu(self.dense2(dense1_output))
        dense3_input = dense2_output
        
    if(self.config["abs_diff"] == False and self.config["cos_sim"] == True):
        dense1_input = torch.cat((query_avg_emb,product_avg_emb),1)
        dense1_output = self.dropout(self.relu(self.dense1(dense1_input)))
        dense2_output = self.relu(self.dense2(dense1_output))
        dense3_input = torch.cat((dense2_output,cos_sim), 1)
        
    if(self.config["abs_diff"] == False and self.config["cos_sim"] == False):
        dense1_input = torch.cat((query_avg_emb,product_avg_emb),1)
        dense1_output = self.dropout(self.relu(self.dense1(dense1_input)))
        dense2_output = self.relu(self.dense2(dense1_output))
        dense3_input = dense2_output
        
    prediction = self.dense3(dense3_input)
    prediction = torch.reshape(prediction,(-1,))
   
    
    return  prediction
