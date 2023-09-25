import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

from torchvision import datasets
from tqdm.notebook import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torchvision.models import ResNet50_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv


class ResNet_baseline(nn.Module):
    def __init__(self, num_diseases=8):
        super(ResNet_baseline, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.fc1 = nn.Linear(2 * 1000, 256)
        self.fc2 = nn.Linear(256, num_diseases)

    def forward(self, img_left, img_right):
        x_left = self.resnet(img_left)
        x_right = self.resnet(img_right)
        x = torch.cat((x_left, x_right), dim =1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        #x = torch.sigmoid(self.fc2(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ResNet_baseline().to(device)
for param in model.resnet.parameters():
   param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")

# from efficientnet paper
loss_weight = torch.tensor([1, 1.2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.2]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)