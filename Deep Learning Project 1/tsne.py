# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 01:19:54 2023

@author: USER
"""

from sklearn.manifold import TSNE
import numpy as np
from Network import one_hot,forward_prop
import matplotlib.pyplot as plt
import pickle

dictionary=np.load("vocab.npy")
model=pickle.load(open("model.pk",'rb'))


embeddings=[]
for i in range(250):
    embeddings.append(np.dot(model[0][0],one_hot(i)))
    
embeddings=np.array(embeddings)
print(embeddings.shape)
embeddings_tsne=TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(embeddings)

x = []
y = []
for embedding in embeddings_tsne:
    x.append(embedding[0])
    y.append(embedding[1])
    
plt.figure(figsize=(16, 16)) 
for i in range(len(x)):
    plt.scatter(x[i],y[i])
    plt.annotate(dictionary[i],
                 xy=(x[i], y[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.show()

custom_data=np.array([[133, 108, 84],[156,200,248],[169,191,248]]) # these are the indices of words "city", "of", "new", "life", 
                                                                   # "in", "the", "he", "is", "the" respectively.
                                                                    

for sentence in custom_data:
    forward=forward_prop(sentence,model[0],model[1])
    prediction_vector=forward[0]
    predicted_label=-1
    prediction_confidence=0
    for j in range(250):
        if prediction_vector[j]>prediction_confidence:
            prediction_confidence=prediction_vector[j]
            predicted_label=j
    print("Prediction for the",dictionary[sentence[0]]+" "+dictionary[sentence[1]]+ " "+dictionary[sentence[2]]+": ",dictionary[predicted_label])