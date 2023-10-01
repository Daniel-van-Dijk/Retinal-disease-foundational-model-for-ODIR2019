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
import datetime
import matplotlib.pyplot as plt
import numpy as np
import cv2


class ODIRDataset(Dataset):
    def __init__(self, dataframe, img_dir, transforms=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transforms = transforms
        

    


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        left_img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Left-Fundus'])
        right_img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Right-Fundus'])
        # print(left_img_name)
        left_image = Image.open(left_img_name)
        right_image = Image.open(right_img_name)
        values = self.dataframe.iloc[idx][5:].values.astype(np.float32)
        labels = torch.tensor(values)

        if self.transforms:
            left_image = self.transforms(left_image)
            right_image = self.transforms(right_image)

        return (left_image, right_image), labels



# croping needs better approach since images have different size and "zoom level"
train_transform = transforms.Compose([               
    transforms.Resize((224, 224)),   
    #transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([      
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# put annotations in current directory
df = pd.read_excel('data/ODIR-5K_Training_Annotations(Updated)_V2.xlsx')
df = df.drop(columns=['Left-Diagnostic Keywords', 'Right-Diagnostic Keywords'])

low_quality_files = [
    "2174_right.jpg",
    "2175_left.jpg",
    "2176_left.jpg",
    "2177_left.jpg",
    "2177_right.jpg",
    "2178_right.jpg",
    "2179_left.jpg",
    "2179_right.jpg",
    "2180_left.jpg",
    "2180_right.jpg",
    "2181_left.jpg",
    "2181_right.jpg",
    "2182_left.jpg",
    "2182_right.jpg",
    "2957_left.jpg",
    "2957_right.jpg"
]
# ALSO ADD THESE to low quality:
# # 2340 left
# 1706_left
# 1710_right
# 4522_left
# 1222_right
# 1260_left
# 2133_right
# 240_left
# 240_right
# 150_left
# 150_right




    
valid_rows = []
for idx, row in df.iterrows():
    left_img_name = os.path.join('data/cropped_ODIR-5K_Training_Dataset', row['Left-Fundus'])
    right_img_name = os.path.join('data/cropped_ODIR-5K_Training_Dataset', row['Right-Fundus'])

    if left_img_name not in low_quality_files and right_img_name not in low_quality_files:
        valid_rows.append(row)

df = pd.DataFrame(valid_rows)

train_df, validation_df = train_test_split(df, test_size=0.10, random_state=42)

print(len(train_df))
# train_dataset = ODIRDataset(train_df, 'data/ODIR-5K_Training_Dataset', transforms=train_transform)
train_dataset = ODIRDataset(train_df, 'data/cropped_ODIR-5K_Training_Dataset', transforms=train_transform)
print(len(train_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

val_dataset = ODIRDataset(validation_df, 'data/cropped_ODIR-5K_Training_Dataset', transforms=val_transform)
print(len(val_dataset))
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
