# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def softmax(x):
    y=np.exp(x)
    z=y/np.sum(y)
    return z

def one_hot(x):
    a=np.zeros(250)
    a[x]=1
    return a

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def cross_entropy(a,b):
    loss=0
    for i in range(len(a)):
        loss=loss-b[i]*np.log(a[i])
    return loss


def forward_prop(x,weights,biases):
   
    
    a_1=np.array([one_hot(x[0]),one_hot(x[1]),one_hot(x[2])])
    a_2=np.array([np.dot(weights[0],one_hot(x[0])),np.dot(weights[0],one_hot(x[1])),np.dot(weights[0],one_hot(x[2]))])
    z_3=np.dot(weights[1],a_2[0])+np.dot(weights[2],a_2[1])+np.dot(weights[3],a_2[2])+biases[0]
    a_3=sigmoid(z_3)
    a_4=np.dot(weights[4],a_3)+biases[1]
    y_hat=softmax(a_4)
    activations=[a_1,a_2,a_3,a_4]
   
    return [y_hat,activations,z_3]

def back_prop(y_hat,target,activations,z_3,weights,biases):
    Local_grad_4=y_hat-target
    Grad_W3=np.outer(Local_grad_4,activations[2])
    Grad_b2=Local_grad_4
    Local_grad_3=np.multiply(np.dot(np.transpose(weights[4]),Local_grad_4),sigmoid_derivative(z_3))
    Grad_W21=np.outer(np.multiply(np.dot(np.transpose(weights[4]),(y_hat-target)),sigmoid_derivative(z_3)),(activations[1][0]))
    Grad_W22=np.outer(np.multiply(np.dot(np.transpose(weights[4]),(y_hat-target)),sigmoid_derivative(z_3)),(activations[1][1]))
    Grad_W23=np.outer(np.multiply(np.dot(np.transpose(weights[4]),(y_hat-target)),sigmoid_derivative(z_3)),(activations[1][2]))     
    Grad_b1=Local_grad_3
    Grad_W1=np.zeros((16,250))
    for i in range(3):
        Grad_W1+=np.outer(np.dot(np.transpose(weights[i+1]),Local_grad_3),(activations[0][i]))
    
    weight_gradients=[Grad_W1,Grad_W21,Grad_W22,Grad_W23,Grad_W3]
    bias_gradients=[Grad_b1,Grad_b2]
    return [weight_gradients,bias_gradients]



