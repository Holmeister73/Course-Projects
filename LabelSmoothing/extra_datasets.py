# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 21:40:48 2024

@author: USER
"""



import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class Cub200Dataset(Dataset):
    def __init__(self, img_path, img_txt, transform=None):
        super(Dataset, self).__init__()
        self.img_path = img_path
        self.transform = transform

        self.img_list = []
        self.targets = []

        with open(img_txt) as f:
            for lines in f:
                _name = (lines.split('\n')[0]).split(';')[0]
                _label = (lines.split('\n')[0]).split(';')[1]
                self.img_list.append(img_path + _name)
                self.targets.append(int(_label))

        self.num_classes = 200
        

    def __getitem__(self, item):
        img = Image.open(self.img_list[item]).convert('RGB')
        label = self.targets[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_list)

class Dataset_with_Indices(Dataset):
    def __init__(self, data):
        self.dst_train = data
    def __getitem__(self, idx):
        
        image = self.dst_train[idx][0]
        label = self.dst_train[idx][1]
        return image, label, idx

    def __len__(self):
        return len(self.dst_train)    
    
