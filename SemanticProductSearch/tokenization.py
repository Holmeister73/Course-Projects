# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:40:54 2024

@author: USER
"""

import numpy as np
import pandas as pd
import re
import pickle
import torch

with open("trigram_vocab_without_stopwords.pk","rb") as filename:
    trigram_vocab = pickle.load(filename)

with open("unigram_vocab_without_stopwords_cleaned_train.pk","rb") as filename:
    unigram_vocab = pickle.load(filename)
    
with open("bigram_vocab_without_stopwords.pk","rb") as filename:
    bigram_vocab = pickle.load(filename)

vocabs = [trigram_vocab,unigram_vocab,bigram_vocab]

stopwords = set()
with open("stopwords.txt","r") as filename:
    lines = filename.readlines()
    for line in lines:
        line = line.strip()
        stopwords.add(line)

#max_query_len = 120  # when all of the tokenization methods are used
#max_product_len = 6000 # when all of the tokenization methods are used
#max_query_len = 70  # with unigram + bigram
#max_product_len = 1500  # with unigram + bigram
#max_query_len = 90 #with char_trigram+unigram
#max_product_len = 5400 #with char_trigram + unigram
max_query_len = 40 # with unigram
max_product_len = 750 # with unigram
#max_query_len = 60 # with char_trigram
#max_product_len = 5000 # with char_trigram



trigram_size = 10000
unigram_size = 15000
bigram_size = 5000


def active_vocab_size(char_trigram = False, unigram = False, bigram = False):
    size = 0
    if(char_trigram): size += trigram_size
    if unigram: size += unigram_size
    if bigram: size += bigram_size
    
    return size

vocab_size = active_vocab_size(char_trigram = False, unigram = True, bigram = False)
oov_size = 45000-vocab_size

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

def numerate_query(token_list, vocabs):
    token_numbers = []
    for i in range(3):
        for token in token_list[i]:
            if token in vocabs[i]:
                token_numbers.append(vocabs[i][token])  
                if(0<vocabs[i][token]<100001 == False):
                    print(vocabs[i][token], token)
            else:
                token_numbers.append(hash(token)%(oov_size)+vocab_size+1)
                if(0< hash(token)%(oov_size)+vocab_size+1 <100001 == False):
                    print(hash(token)%(oov_size)+vocab_size+1, token)
    for i in range(len(token_numbers),max_query_len):
        token_numbers.append(0)
    return torch.IntTensor(token_numbers)      
      
def token_length(token_list):
    if len(token_list[0])+len(token_list[1])+len(token_list[2])>0:
        return len(token_list[0])+len(token_list[1])+len(token_list[2])
    else: return 1
    
    
def numerate_product(token_list, vocabs):
    token_numbers = []
    for i in range(3):
        for token in token_list[i]:
            if token in vocabs[i]:
                token_numbers.append(vocabs[i][token])
                if(0<vocabs[i][token]<100001 == False):
                    print(vocabs[i][token], token)
            else:
                token_numbers.append(hash(token)%(oov_size)+vocab_size+1)
                if(0< hash(token)%(oov_size)+vocab_size+1 <100001 == False):
                    print(hash(token)%(oov_size)+vocab_size+1, token)
    for i in range(len(token_numbers),max_product_len):
        token_numbers.append(0)
    return torch.IntTensor(token_numbers) 

def numerated_df():
    df_train = pd.read_csv("cleaned_train.csv",encoding = "ISO-8859-1")
    df_valid = pd.read_csv("cleaned_validation.csv",encoding = "ISO-8859-1")
    df_test = pd.read_csv("cleaned_test.csv",encoding = "ISO-8859-1")
    
    df_train["tokenized_product"] = df_train["cleaned_product"].apply(lambda x: tokenize(x,char_trigram = False, unigram = True, bigram = False))
    df_train["tokenized_query"] = df_train["corrected_query"].apply(lambda x:tokenize(x, char_trigram = False, unigram = True, bigram = False))
    
    df_valid["tokenized_product"] = df_valid["corrected_query"].apply(lambda x: tokenize(x,char_trigram = False, unigram = True, bigram = False))
    df_valid["tokenized_query"] = df_valid["corrected_query"].apply(lambda x:tokenize(x, char_trigram = False, unigram = True, bigram = False))
    
    df_test["tokenized_product"] = df_test["cleaned_product"].apply(lambda x: tokenize(x,char_trigram = False, unigram = True, bigram = False))
    df_test["tokenized_query"] = df_test["corrected_query"].apply(lambda x:tokenize(x, char_trigram = False, unigram = True, bigram = False))
    
    df_train["numerated_product"] = df_train["tokenized_product"].apply(lambda x: numerate_product(x,vocabs))
    df_train["numerated_query"] = df_train["tokenized_query"].apply(lambda x: numerate_query(x,vocabs))
    
    df_valid["numerated_product"] = df_valid["tokenized_product"].apply(lambda x: numerate_product(x, vocabs))
    df_valid["numerated_query"] = df_valid["tokenized_query"].apply(lambda x: numerate_query(x,vocabs))
    
    df_test["numerated_product"] = df_test["tokenized_product"].apply(lambda x: numerate_product(x, vocabs))
    df_test["numerated_query"] = df_test["tokenized_query"].apply(lambda x: numerate_query(x,vocabs))
    
    df_train["product_length"] = df_train["tokenized_product"].apply(token_length)
    df_train["query_length"] = df_train["tokenized_query"].apply(token_length)
   
    df_valid["product_length"] = df_valid["tokenized_product"].apply(token_length)
    df_valid["query_length"] = df_valid["tokenized_query"].apply(token_length)
    
    df_test["product_length"] = df_test["tokenized_product"].apply(token_length)
    df_test["query_length"] = df_test["tokenized_query"].apply(token_length)
    
    
    return df_train, df_valid, df_test





