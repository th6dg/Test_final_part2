import pandas as pd
import numpy as np
import torch
from utils.open_function import open_images_dataset, open_labels_dataset

class Mnist_Train():
    def __init__(self):
        # open dataset
        self.images_train = open_images_dataset('./MNIST/train-images.idx3-ubyte')
        self.labels_train = open_labels_dataset('./MNIST/train-labels.idx1-ubyte')
    
    def __getitem__(self, index):
        image = np.array(self.images_train[index])
        label = np.array(self.labels_train[index])
        return torch.reshape(torch.from_numpy(image).float(),(28,28)).unsqueeze_(0), torch.from_numpy(label).long()
    
    def __len__(self):
        return len(self.labels_train)
    
class Mnist_Test():
    def __init__(self):
        # open dataset
        self.images_train = open_images_dataset('./MNIST/t10k-images.idx3-ubyte')
        self.labels_train = open_labels_dataset('./MNIST/t10k-labels.idx1-ubyte')
    
    def __getitem__(self, index):
        image = np.array(self.images_train[index])
        label = np.array(self.labels_train[index])
        return torch.reshape(torch.from_numpy(image).float(),(28,28)).unsqueeze_(0), torch.from_numpy(label)
    
    def __len__(self):
        return len(self.labels_train)
    
