# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np


def build_dataset(is_train, args):
    
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train == 'train':
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
    ])
    else:
        
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    return transform


class ODIRDataset(Dataset):
    def __init__(self, dataframe, img_dir, is_train, args):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.is_train = is_train
        self.transforms = build_transform(is_train, args)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        left_img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Left-Fundus'])
        right_img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Right-Fundus'])

        left_image = Image.open(left_img_name)
        right_image = Image.open(right_img_name)

        values = self.dataframe.iloc[idx][5:].values.astype(np.float32)
        labels = torch.tensor(values)

        if self.transforms:
            left_image = self.transforms(left_image)
            right_image = self.transforms(right_image)

        return (left_image, right_image), labels
    

class TestDataset(Dataset):
    def __init__(self, folder_path, is_train, args):
        self.folder_path = folder_path
        self.transform = build_transform(is_train, args)
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files) // 2

    def __getitem__(self, idx):
        left_image_name = self.image_files[2 * idx]
        right_image_name = self.image_files[2 * idx + 1]

        image_id = int(left_image_name.split('_')[0])

        left_image = Image.open(os.path.join(self.folder_path, left_image_name))
        right_image = Image.open(os.path.join(self.folder_path, right_image_name))

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return left_image, right_image, image_id
