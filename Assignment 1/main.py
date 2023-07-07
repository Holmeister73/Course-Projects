# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:41:53 2023

@author: USER
"""

import numpy as np
from Network import forward_prop,back_prop,one_hot,cross_entropy
import pickle

train_data=np.load("train_inputs.npy")
train_labels=np.load("train_targets.npy")
validation_data=np.load("valid_inputs.npy")
validation_labels=np.load("valid_targets.npy")
copy_data=np.copy(train_data)
copy_labels=np.copy(train_labels)

def validation_accuracy(weights,biases):
    matches=0
    loss=0
    for i in range(len(validation_data)):
        forward=forward_prop(validation_data[i],weights,biases)
        prediction_vector=forward[0]
        loss+=cross_entropy(prediction_vector,one_hot(validation_labels[i]))
        predicted_label=-1
        prediction_confidence=0
        for j in range(250):
            if prediction_vector[j]>prediction_confidence:
                prediction_confidence=prediction_vector[j]
                predicted_label=j
        if(predicted_label==validation_labels[i]):
            matches+=1
    return [matches/len(validation_data),loss/len(validation_data)]

def training_accuracy(weights,biases):
    matches=0
    loss=0
    for i in range(len(train_data)):
        forward=forward_prop(train_data[i],weights,biases)
        prediction_vector=forward[0]
        loss+=cross_entropy(prediction_vector,one_hot(train_labels[i]))
        predicted_label=-1
        prediction_confidence=0
        for j in range(250):
            if prediction_vector[j]>prediction_confidence:
                prediction_confidence=prediction_vector[j]
                predicted_label=j
        if(predicted_label==train_labels[i]):
            matches+=1
    return [matches/len(train_data),loss/len(train_data)]


def train():
    mu=0
    sigma=0.01
    eta=0.01
    batchsize=50
    W_1=np.random.normal(mu,sigma,(16,250))
    W_21=np.random.normal(mu,sigma,(128,16))
    W_22=np.random.normal(mu,sigma,(128,16))
    W_23=np.random.normal(mu,sigma,(128,16))
    W_3=np.random.normal(mu,sigma,(250,128))
    b_1=np.zeros(128)
    b_2=np.zeros(250)
    
    for i in range(15):
        permutation=np.random.permutation(len(train_data))
        shuffled_data=copy_data[permutation]
        shuffled_labels=copy_labels[permutation]
        minibatches=np.split(shuffled_data,len(train_data)/batchsize)
        minibatch_labels=np.split(shuffled_labels,len(train_data)/batchsize)
        for j in range(int(len(train_data)/batchsize)):
            backward=[[np.zeros((16,250)),np.zeros((128,16)),np.zeros((128,16)),np.zeros((128,16)),np.zeros((250,128))],[np.zeros(128),np.zeros(250)]]
            weights=[W_1,W_21,W_22,W_23,W_3]
            biases=[b_1,b_2]
            for k in range(batchsize):
                forward=forward_prop(minibatches[j][k],weights,biases)
                gradient=back_prop(forward[0],one_hot(minibatch_labels[j][k]),forward[1],forward[2],weights,biases)
                backward[0][0]+=gradient[0][0]
                backward[0][1]+=gradient[0][1]
                backward[0][2]+=gradient[0][2]
                backward[0][3]+=gradient[0][3]
                backward[0][4]+=gradient[0][4]
                backward[1][0]+=gradient[1][0]
                backward[1][1]+=gradient[1][1]
            
            W_1=W_1-eta*backward[0][0]
            W_21=W_21-eta*backward[0][1]
            W_22=W_22-eta*backward[0][2]
            W_23=W_23-eta*backward[0][3]
            W_3=W_3-eta*backward[0][4]
            b_1=b_1-eta*backward[1][0]
            b_2=b_2-eta*backward[1][1]
           
        
        print("Validation accuracy and validation loss: ",validation_accuracy(weights,biases))
       
    print("Training accuracy and training loss: ",training_accuracy(weights,biases))
    
    filename=open("model.pk",'wb')
    pickle.dump([weights,biases],filename)        
   
train()  
    
