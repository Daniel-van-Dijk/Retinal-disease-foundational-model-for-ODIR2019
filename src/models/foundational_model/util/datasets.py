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
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import pandas as pd

def build_dataset(is_train, args):
    
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset

def paired_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    # Train transform
    if is_train == 'train':
        basic_transforms = [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        
        random_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-15, 15)),
            # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.RandomResizedCrop(size=224, scale=(0.85, 1.15)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
        
        return random_transforms, basic_transforms
    else:
        return None, [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

class ODIRDataset2eye(Dataset):
    def __init__(self, dataframe, img_dir, is_train, args):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.is_train = is_train
        self.random_transforms, self.basic_transforms = paired_transform(is_train, args)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        left_img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Left-Fundus'])
        right_img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Right-Fundus'])

        left_image = Image.open(left_img_name)
        right_image = Image.open(right_img_name)

        values = self.dataframe.iloc[idx][5:].values.astype(np.float32)
        labels = torch.tensor(values)

        seed = torch.randint(0, 2**32, (1,)).item()

        if self.random_transforms:
            random_transform = transforms.Compose(self.random_transforms)
            torch.manual_seed(seed)  
            left_image = random_transform(left_image)
            torch.manual_seed(seed)  
            right_image = random_transform(right_image)

        basic_transform = transforms.Compose(self.basic_transforms)
        left_image = basic_transform(left_image)
        right_image = basic_transform(right_image)

        return (left_image, right_image), labels


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train == 'train':
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomResizedCrop(size=224, scale=(0.85, 1.15)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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


class Merge_valset(Dataset):
    def __init__(self, dataframe, img_dir, is_train, args):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.is_train = is_train
        self.transforms = build_transform(is_train, args)

    def __len__(self):
        return len(self.dataframe) // 2 

    def __getitem__(self, idx):
        idx *= 2 

        left_img_path = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Left-Fundus'])
        right_img_path = os.path.join(self.img_dir, self.dataframe.iloc[idx + 1]['Right-Fundus'])

        left_image = Image.open(left_img_path)
        right_image = Image.open(right_img_path)

        left_labels = torch.tensor(self.dataframe.iloc[idx][5:].values.astype(np.float32))
        right_labels = torch.tensor(self.dataframe.iloc[idx + 1][5:].values.astype(np.float32))
        left_image_id = int(self.dataframe.iloc[idx]['Left-Fundus'].split('_')[0])
        right_image_id = int(self.dataframe.iloc[idx + 1]['Right-Fundus'].split('_')[0])

        assert left_image_id == right_image_id

        if self.transforms:
            left_image = self.transforms(left_image)
            right_image = self.transforms(right_image)

        return (left_image, right_image, left_image_id)




class ODIRDataset(Dataset):
    def __init__(self, dataframe, img_dir, is_train, args):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.is_train = is_train
        self.transforms = build_transform(is_train, args)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if not pd.isna(self.dataframe.iloc[idx]['Left-Fundus']):
            img = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Left-Fundus'])
        if not pd.isna(self.dataframe.iloc[idx]['Right-Fundus']):
            img = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Right-Fundus'])

        image = Image.open(img)
        values = self.dataframe.iloc[idx][5:].values.astype(np.float32)
        labels = torch.tensor(values)

        if self.transforms:
            image = self.transforms(image)

        return image, labels
    

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

        return (left_image, right_image, image_id)
    


def save_batch_images(dataloader, num_images=8, filename="batch_visualization.png"):
    data_iter = iter(dataloader)
    (left_images, right_images), labels = next(data_iter)  
    concatenated_images = torch.cat((left_images, right_images), 0)[:2*num_images] 
    print(left_images.shape, right_images.shape)
    plt.figure(figsize=(15, 7))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(concatenated_images, padding=2, normalize=True).cpu(), (1, 2, 0)))
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
