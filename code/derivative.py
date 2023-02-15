import sys
sys.path.append('./code')
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import time
import os
import copy
import random
import argparse
from vgg_face import *
from FAUDataset import *


num_classes = 10
batch_size = 32
checkpoint_path = 'D:/FAU_models/models_r16/checkpoint_epoch49.pth'
dataset_root = 'D:/Datasets/FAU'

data_transforms = {
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
    }

def set_parameter_requires_grad(model, feature_extracting):
    for param in model.parameters():
        param.requires_grad = False if feature_extracting else True

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

model = VGG_16()
set_parameter_requires_grad(model, feature_extracting=False)
num_ftrs = model.fc8.in_features
model.fc8 = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(checkpoint_path), strict=False)

input = 0
output = model(input)

output.backward(retain_graph=True)
feature_derivative = input.grad

print(feature_derivative.shape)