# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import spacy
import string
import gensim
import operator
import re
import chardet
import pickle
import torch
"""
Created on Sat Oct  7 18:37:34 2023

@author: USER
"""

stopwords = set()
with open("stopwords.txt","r") as filename:
    lines = filename.readlines()
    for line in lines:
        line = line.strip()
        stopwords.add(line)
        
        
def tokenize(text,char_trigram = False, unigram = False, bigram = False):
    char_trigrams = []
    unigrams = []
    bigrams = []
    words_with_stopwords = text.split(" ")
    words = []
    for word in words_with_stopwords:
        if word not in stopwords:
            words.append(word)
    new_text = " ".join(words)
    
    if char_trigram == True:
        for i in range(len(new_text)-2):
            token = new_text[i:i+3].lower()
            char_trigrams.append(token)

    if unigram == True:
        for word in words:
            unigrams.append(word.lower())
    if bigram == True:
        for i in range(len(words)-1):
            bigrams.append(words[i].lower()+" "+words[i+1].lower())
    
    token_list = [char_trigrams,unigrams,bigrams]
    return token_list

def token_length(token_list):
    return len(token_list[0])+len(token_list[1])+len(token_list[2])
    
    


df_train = pd.read_csv("train.csv",encoding = "ISO-8859-1")
df_products = pd.read_csv("product_descriptions.csv", encoding = "ISO-8859-1")

df_train = pd.merge(df_train, df_products)

df_train["product"]=df_train["product_title"]+" "+df_train["product_description"]
df_train = df_train.drop(['product_title', 'product_description'], axis = 1)

df_train["tokenized_product"] = df_train["product"].apply(lambda x: tokenize(x,bigram = True))
df_train["tokenized_query"] = df_train["search_term"].apply(lambda x: tokenize(x,bigram = True))

df_train["product_length"] = df_train["tokenized_product"].apply(token_length)
df_train["query_length"] = df_train["tokenized_query"].apply(token_length)
df_train = df_train.drop(["product", "search_term"], axis = 1)

df_valid = df_train.sample(frac = 0.1, random_state = 42)      
df_train = df_train.drop(df_valid.index)
df_test = df_train.sample(frac = 0.11, random_state = 42)
df_train = df_train.drop(df_test.index)

"""trigram_freqs = {}

for token_list in list(df_train["tokenized_product"]):
    for token in token_list[0]:
        if token not in trigram_freqs:
            trigram_freqs[token] = 1
        else:
            trigram_freqs[token]+= 1

for token_list in list(df_train["tokenized_query"]):
    for token in token_list[0]:
        if token not in trigram_freqs:
            trigram_freqs[token] = 1
        else:
            trigram_freqs[token]+= 1
            
print(len(trigram_freqs))



trigram_freqs_sorted = dict(sorted(trigram_freqs.items(), key=lambda x:x[1], reverse = True))





trigram_vocab = {}
token_counter = 1
for key,val in trigram_freqs_sorted.items():
    trigram_vocab[key] = token_counter
    token_counter+= 1
    if token_counter == 10001:
        break

print(list(trigram_vocab.items())[-10:],len(trigram_vocab))


with open("trigram_vocab_without_stopwords.pk",'wb') as filename:
    pickle.dump(trigram_vocab,filename)   """
"""unigram_freqs = {}

for token_list in list(df_train["tokenized_product"]):
    for token in token_list[1]:
        if token not in unigram_freqs:
            unigram_freqs[token] = 1
        else:
            unigram_freqs[token]+= 1

for token_list in list(df_train["tokenized_query"]):
    for token in token_list[1]:
        if token not in unigram_freqs:
            unigram_freqs[token] = 1
        else:
            unigram_freqs[token]+= 1
            
print(len(unigram_freqs))



unigram_freqs_sorted = dict(sorted(unigram_freqs.items(), key=lambda x:x[1], reverse = True))





unigram_vocab = {}
token_counter = 10001
for key,val in unigram_freqs_sorted.items():
    unigram_vocab[key] = token_counter
    token_counter+= 1
    if(token_counter == 30001):
        break

print(list(unigram_vocab.items())[:10],len(unigram_vocab))


with open("unigram_vocab_without_stopwords.pk",'wb') as filename:
    pickle.dump(unigram_vocab,filename)   """
    
bigram_freqs = {}

for token_list in list(df_train["tokenized_product"]):
    for token in token_list[2]:
        if token not in bigram_freqs:
            bigram_freqs[token] = 1
        else:
            bigram_freqs[token]+= 1

for token_list in list(df_train["tokenized_query"]):
    for token in token_list[2]:
        if token not in bigram_freqs:
            bigram_freqs[token] = 1
        else:
            bigram_freqs[token]+= 1
            
print(len(bigram_freqs))



bigram_freqs_sorted = dict(sorted(bigram_freqs.items(), key=lambda x:x[1], reverse = True))

bigram_vocab = {}
token_counter = 30001
for key,val in bigram_freqs_sorted.items():
    bigram_vocab[key] = token_counter
    token_counter+= 1
    if(token_counter == 35001):
        break
print(list(bigram_vocab.items())[-10:],len(bigram_vocab))


with open("bigram_vocab_without_stopwords.pk",'wb') as filename:
    pickle.dump(bigram_vocab,filename)    