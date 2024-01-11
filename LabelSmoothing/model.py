# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 15:04:54 2023

@author: USER
"""


"""

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
    """

import torch
import torch.nn as nn
import torch.nn.functional as F


    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, dataset="cifar10"):
        
        super(ResNet, self).__init__()
        self.dataset = dataset
        if(dataset == "cifar10"):
            num_classes = 10
        if(dataset == "cifar100"):
            num_classes = 100
        
        if(dataset.startswith("cifar")):
            self.in_planes = 64
    
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.fc = nn.Linear(512*block.expansion, num_classes)
        
        
        if(dataset == "cub200"):
            num_classes = 200
            
            self.in_planes = 64
    
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.avgpool = nn.AvgPool2d(7) 
            self.fc = nn.Linear(512*block.expansion, num_classes)
            
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if(self.dataset.startswith("cifar")):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            emb = out
            out = self.fc(out)
            
        if(self.dataset == "cub200"):
           
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            emb = out
            out = self.fc(out) 
        
        
        
        return out, emb


class ResNet_Pencil(nn.Module):
    def __init__(self, block, num_blocks, original_labels, dataset = "cifar10"):
        super(ResNet_Pencil, self).__init__()
        
        
        self.dataset = dataset
        if(dataset == "cifar10"):
            num_classes = 10
        if(dataset == "cifar100"):
            num_classes = 100
        
        if(dataset.startswith("cifar")):
            self.in_planes = 64
    
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.fc = nn.Linear(512*block.expansion, num_classes)
            self.learned_labels = nn.Embedding(original_labels.size(0), original_labels.size(1))
            self.learned_labels.weight.data += 10*original_labels
            self.make_label = nn.Softmax(dim = 1)
            self.make_log_label = nn.LogSoftmax(dim = 1)
        if(dataset == "cub200"):
            num_classes = 200
            
            self.in_planes = 64
    
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.avgpool = nn.AvgPool2d(7) 
            self.fc = nn.Linear(512*block.expansion, num_classes)
            self.learned_labels = nn.Embedding(original_labels.size(0), original_labels.size(1))
            self.learned_labels.weight.data += 10*original_labels
            self.make_label = nn.Softmax(dim = 1)
            self.make_log_label = nn.LogSoftmax(dim = 1)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def get_learnt_labels(self, indices):
        return self.learned_labels(torch.LongTensor([0]).to("cuda"))
    
    def forward(self, x, indices): #indices will be torch IntTensor
    
        if(self.dataset.startswith("cifar")):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            learned_labels = self.learned_labels(indices)
            log_labels = self.make_log_label(self.learned_labels(indices))
            
            
        if(self.dataset == "cub200"):
           
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            learned_labels = self.learned_labels(indices)
            log_labels = self.make_log_label(self.learned_labels(indices))
            
        return out, learned_labels, log_labels



def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet18_Pencil(original_labels, num_classes = 10):
    return ResNet_Pencil(BasicBlock, [2, 2, 2, 2], original_labels, num_classes = num_classes)

def ResNet34(dataset):
    return ResNet(BasicBlock, [3, 4, 6, 3], dataset = dataset)

def ResNet34_Pencil(dataset, original_labels):
    return ResNet_Pencil(BasicBlock, [3, 4, 6, 3], original_labels, dataset = dataset)

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

    