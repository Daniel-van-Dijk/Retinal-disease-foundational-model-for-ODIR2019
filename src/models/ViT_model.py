import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
import os
from torchvision import datasets
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
import timm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import numpy as np
import datetime


class VisionTransformer(nn.Module):
    def __init__(self, num_classes=8):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes

        # Vision Transformer backbone
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Replace the final classification layer
        self.backbone.head = nn.Identity()

        # Add a classification head for multi-label classification
        self.classification_head = nn.Linear(self.backbone.embed_dim * 2, num_classes)


    def forward(self, image_left, image_right):
        features_left = self.backbone(image_left)
        features_right = self.backbone(image_right)
        combined_features = torch.cat((features_left, features_right), dim=1)

        # Pass through the classification head
        logits = self.classification_head(combined_features)

        return logits

        # Pass through the classification head
        logits = self.classification_head(combined_features)

        return logits
