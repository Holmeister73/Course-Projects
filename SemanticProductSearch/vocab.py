# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 09:52:17 2024

@author: USER
"""

import numpy as np
import pandas as pd
import re
import pickle
import torch
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))


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
    
    
df_train = pd.read_csv("cleaned_train.csv", encoding = "ISO-8859-1")
df_train["tokenized_product"] = df_train["cleaned_product"].apply(lambda x: tokenize(x, unigram = True))
df_train["tokenized_query"] = df_train["corrected_query"].apply(lambda x: tokenize(x, unigram = True))

unigram_freqs = {}

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
token_counter = 1
for key,val in unigram_freqs_sorted.items():
    unigram_vocab[key] = token_counter
    token_counter+= 1
    if(token_counter == 15001):
        break

print(list(unigram_vocab.items())[:10],len(unigram_vocab))


with open("unigram_vocab_without_stopwords_cleaned_train.pk",'wb') as filename:
    pickle.dump(unigram_vocab,filename)
    
