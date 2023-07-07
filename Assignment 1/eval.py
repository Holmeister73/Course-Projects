# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:52:11 2023

@author: USER
"""

import numpy as np
from Network import forward_prop,cross_entropy,one_hot
import pickle

test_data=np.load("test_inputs.npy")
test_labels=np.load("test_targets.npy")
model=pickle.load(open("model.pk",'rb'))


matches=0
loss=0

for i in range(len(test_data)):
    forward=forward_prop(test_data[i],model[0],model[1])
    prediction_vector=forward[0]
    loss+=cross_entropy(prediction_vector,one_hot(test_labels[i]))
    predicted_label=-1
    prediction_confidence=0
    for j in range(250):
        if prediction_vector[j]>prediction_confidence:
            prediction_confidence=prediction_vector[j]
            predicted_label=j
    if(predicted_label==test_labels[i]):
        matches+=1
        
print( "Test accuracy and test loss =",[matches/len(test_data),loss/len(test_data)])

