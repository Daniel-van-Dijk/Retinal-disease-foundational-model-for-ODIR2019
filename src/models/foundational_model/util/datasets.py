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


# def build_transform(is_train, args):
#     mean = IMAGENET_DEFAULT_MEAN
#     std = IMAGENET_DEFAULT_STD
#     # train transform
#     if is_train=='train':
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=args.color_jitter,
#             auto_augment=args.aa,
#             interpolation='bicubic',
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
#             mean=mean,
#             std=std,
#         )
#         return transform

#     # eval transform
#     t = []
#     if args.input_size <= 224:
#         crop_pct = 224 / 256
#     else:
#         crop_pct = 1.0
#     size = int(args.input_size / crop_pct)
#     t.append(
#         transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
#     )
#     t.append(transforms.CenterCrop(args.input_size))
#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(mean, std))
#     return transforms.Compose(t)

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train':
     #
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(),  
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
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

def custom_transform(input_size, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, is_training=False):
    if is_training:
        # Define your training transforms here. E.g., resizing, normalization, and any other data augmentations.
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),  # optional, if you want to include random horizontal flip in training
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        # Define your evaluation transforms here. E.g., resizing and normalization.
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
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