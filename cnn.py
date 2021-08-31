import os
import random

import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from collections import OrderedDict



#下载数据集
def load_CIFAR(path='./datasets/'):
  NUM_TRAIN = 49000
  # The torchvision.transforms package provides tools for preprocessing data
  # and for performing data augmentation; here we set up a transform to
  # preprocess the data by subtracting the mean RGB value and dividing by the
  # standard deviation of each RGB value; we've hardcoded the mean and std.
  transform = T.Compose([
                  T.ToTensor(),
                  T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
              ])

  # We set up a Dataset object for each split (train / val / test); Datasets load
  # training examples one at a time, so we wrap each Dataset in a DataLoader which
  # iterates through the Dataset and forms minibatches. We divide the CIFAR-10
  # training set into train and val sets by passing a Sampler object to the
  # DataLoader telling how it should sample from the underlying Dataset.
  cifar10_train = dset.CIFAR10(path, train=True, download=True,
                               transform=transform)
  loader_train = DataLoader(cifar10_train, batch_size=64, 
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

  cifar10_val = dset.CIFAR10(path, train=True, download=True,
                             transform=transform)
  loader_val = DataLoader(cifar10_val, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

  cifar10_test = dset.CIFAR10(path, train=False, download=True, 
                              transform=transform)
  loader_test = DataLoader(cifar10_test, batch_size=64)
  return loader_train, loader_val, loader_test

#定义flatten
def flatten(x, start_dim=1, end_dim=-1):
  return x.flatten(start_dim=start_dim, end_dim=end_dim)

class Flatten(nn.Module):
  def forward(self, x):
    return flatten(x)

#初始化网络
def initialize_cnn():
 
  C, H, W = 3, 16, 16
  num_classes1 = 1024
  num_classes2 = 10

  # Hidden layer channel and kernel sizes
  channel_1 = 32
  channel_2 = 32
  channel_3 = 16
  kernel_size_1 = 3
  pad_size_1 = 1
  kernel_size_2 = 3
  pad_size_2 = 1
  kernel_size_3 = 3
  pad_size_3 = 1
  kernel_size_4 = 2

  # Parameters for optimizer
  learning_rate = 1e-2
  weight_decay = 1e-4
  momentum = 0.5

  model = None
  optimizer = None
  ##################################################################################
  #   1. Convolutional layer (with bias) with 32 3x3 filters, with zero-padding of 1 
  #   2. ReLU                                      
  #   3. Convolutional layer (with bias) with 32 3x3 filters, with zero-padding of 1 
  #   4. ReLU                              
  #   5. Convolutional layer (with bias) with 16 3x3 filters, with zero-padding of 1 
  #   6. BatchNorm
  #   7. ReLU
  #   8. pool
  #   9. Fully-connected layer (with bias) to compute scores for 128 classes     
  #  10. Fully-connected layer (with bias) to compute scores for 10 classes     
  ##################################################################################                                       
  model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(C, channel_1, kernel_size_1, padding=pad_size_1)),
    ('relu1', nn.ReLU()),
    ('conv2', nn.Conv2d(channel_1, channel_2, kernel_size_2, padding=pad_size_2)),
    ('relu2', nn.ReLU()),
    ('conv3', nn.Conv2d(channel_2, channel_3, kernel_size_3, padding=pad_size_3)),
    ('BN',nn.BatchNorm2d(channel_3)),
    ('relu3', nn.ReLU()),
    ('pool',nn.MaxPool2d(kernel_size_4,stride=kernel_size_4)),
    ('flatten', Flatten()),
    ('fc1', nn.Linear(channel_3*H*W, num_classes1)),
    ('fc2', nn.Linear(num_classes1, num_classes2)),
  ]))
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                        weight_decay=weight_decay,
                        momentum=momentum, nesterov=True)
  ################################################################################
  #                                 END OF YOUR CODE                             
  ################################################################################
  return model, optimizer