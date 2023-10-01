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


def fov_extraction(image_path, image_name, cropped_dir):
        # Read the image
        img = cv2.imread(image_path)
        # RGB -> BGR
        # Extract the red channel
        red_channel = img[:,:,2]

        # Compute the centerlines
        h, w = red_channel.shape
        Hcenterline = h // 2
        Vcenterline = w // 2
        print(Hcenterline)
        # Extract intensity profiles along the centerlines
        horizontal_profile = red_channel[Hcenterline, :]
        vertical_profile = red_channel[:, Vcenterline]
        
        # Compute threshold from the horizontal profile
        Hthreshold = np.max(horizontal_profile) * 0.06
        Vthreshold = np.max(vertical_profile) * 0.06
        
        # Identify transitions based on the threshold
        # transitions_horizontal = np.where(np.diff((horizontal_profile > threshold).astype(int)) != 0)[0]
        # transitions_vertical = np.where(np.diff((vertical_profile > threshold).astype(int)) != 0)[0]
        binary_horizontal_profile = (horizontal_profile > Hthreshold).astype(int)
        binary_vertical_profile = (vertical_profile > Vthreshold).astype(int)

        diff_horizontal = np.diff(binary_horizontal_profile)
        diff_vertical = np.diff(binary_vertical_profile)

        transitions_horizontal = np.where(diff_horizontal != 0)[0]
        transitions_vertical = np.where(diff_vertical != 0)[0]

        
        # fix for no vertical / horizontal transition because of too much zoom 
        # we use original width / height of image
        if len(transitions_horizontal) < 2:
            print("Could not find horizontal transitions for", image_path)
            print("Using original width of image so no cropping applied")
            transitions_horizontal = [0, img.shape[1]-1]
        
        if len(transitions_vertical) < 2:
            print("Could not find vertical transitions for", image_path)
            print("Using original height of image so no cropping applied")
            transitions_vertical = [0, img.shape[0]-1]

        vertical_diff = transitions_vertical[-1] - transitions_vertical[0]
        horizontal_diff = transitions_horizontal[-1] - transitions_horizontal[0]
        
        # if horizontal_diff < img.shape[1] * 0.25 or horizontal_diff < vertical_diff * 0.9:
        if horizontal_diff < img.shape[1] * 0.25:
            print("horizontal diff: ", horizontal_diff, "vertical diff: ", vertical_diff)
            print('image size: ', img.shape)

            print("Wrong transition detected so no cropping for ", image_path)
            transitions_horizontal = [0, img.shape[1]-1]

        if vertical_diff < img.shape[0] * 0.25:
            print("horizontal diff: ", horizontal_diff, "vertical diff: ", vertical_diff)
            print('image size: ', img.shape)
            print("Wrong vertical transition detected so no cropping for ", image_path)
            transitions_vertical = [0, img.shape[0]-1]

        if len(transitions_vertical) > 2:
            print("More than 2 vertical transitions for", image_path)
            print(transitions_vertical)

        
        if len(transitions_horizontal) > 2:
            print("More than 2 horizontal transitions for", image_path)
            print(transitions_horizontal)
    
    
        
        # Crop image based on the transitions
        cropped_img = img[transitions_vertical[0]:transitions_vertical[-1], transitions_horizontal[0]:transitions_horizontal[-1]]
        full_path = os.path.join(cropped_dir, image_name)
        # print('full path', full_path)
        cv2.imwrite(full_path, cropped_img)
        # Convert to PIL image for compatibility with torchvision transforms
        result = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

        return result


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
    transforms.Resize((256, 256)),        
    transforms.CenterCrop(200),          
    transforms.Resize((224, 224)),   
    #transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),        
    transforms.CenterCrop(200),          
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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



# import shutil

# for img in shaded_imgs:
#     src_path = os.path.join('data/ODIR-5K_Training_Dataset', img)
    
#     # Check if the file exists in the source directory
#     if os.path.exists(src_path):
#         dest_path = os.path.join('data/shaded_images', img)
#         shutil.copy2(src_path, dest_path)
#     else:
#         print(f"Warning: {img} not found in {'data/shaded_images'}")

# print("Copying process completed!")



# valid_rows = []
# for idx, row in df.iterrows():
#     # left_img_name = os.path.join('data/ODIR-5K_Training_Dataset', row['Left-Fundus'])
#     # right_img_name = os.path.join('data/ODIR-5K_Training_Dataset', row['Right-Fundus'])

#     if row['Left-Fundus'] in shaded_imgs and row['Right-Fundus'] in shaded_imgs:
#         left_img_name = os.path.join('data/shaded_images', row['Left-Fundus'])
#         right_img_name = os.path.join('data/shaded_images', row['Right-Fundus'])
#         print(left_img_name)
#         print(row['Left-Fundus'])
        
#         left_img_check = fov_extraction(left_img_name, row['Left-Fundus'], 'data/fixed_shaded')
#         right_img_check = fov_extraction(right_img_name, row['Right-Fundus'], 'data/fixed_shaded')
        
#         if left_img_check is not None and right_img_check is not None and left_img_name not in low_quality_files and right_img_name not in low_quality_files:
#             valid_rows.append(row)

# df = pd.DataFrame(valid_rows)

import time

# Define a function to measure execution time
def crop_images(df):
    start_time = time.time()
    
    valid_rows = []
    for idx, row in df.iterrows():
        left_img_name = os.path.join('data/ODIR-5K_Training_Dataset', row['Left-Fundus'])
        right_img_name = os.path.join('data/ODIR-5K_Training_Dataset', row['Right-Fundus'])

        left_img_check = fov_extraction(left_img_name, row['Left-Fundus'], 'data/cropped_images_new')
        right_img_check = fov_extraction(right_img_name, row['Right-Fundus'], 'data/cropped_images_new')
    
        if left_img_name not in low_quality_files and right_img_name not in low_quality_files:
            valid_rows.append(row)

    df = pd.DataFrame(valid_rows)

    train_df, validation_df = train_test_split(df, test_size=0.10, random_state=42)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    return train_df, validation_df

train_df, valiation_df = crop_images(df)

# 232_left.jpg 232_right.jpg 




# train_df = oversample_minority(train_df)
print(len(train_df))
# train_dataset = ODIRDataset(train_df, 'data/ODIR-5K_Training_Dataset', transforms=train_transform)
train_dataset = ODIRDataset(train_df, 'data/cropped_images')
print(len(train_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
print(len(train_dataloader))
print('got here')
exit()
with torch.no_grad():
    for (images_left, images_right), labels in tqdm(train_dataloader):
        print(images_left)


def visualize_random_samples(dataset, num_samples=5, filename="output.png"):
    """
    Visualize random sample pairs (left and right images) from the dataset and save the visualization to a file.
    
    Parameters:
    - dataset: The dataset object from which samples will be drawn.
    - num_samples: Number of random samples to display.
    - filename: Name of the output file where visualization will be saved.
    """
    # Randomly sample indices from the dataset
    indices = np.random.choice(len(dataset), num_samples)
    
    plt.figure(figsize=(15,5))
    
    for i, idx in enumerate(indices, 1):
        (left_img, right_img), labels = dataset[idx]
        
        # Display left image
        plt.subplot(2, num_samples, i)
        show_image(left_img)
        plt.title(f'Sample {i} - Left')
        
        # Display right image
        plt.subplot(2, num_samples, i + num_samples)
        show_image(right_img)
        plt.title(f'Sample {i} - Right')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')

# Assuming you have already defined and populated `train_dataset`
#visualize_random_samples(train_dataset, filename="sample_visualization3.png")

def get_unique_image_sizes(img_dir):
    unique_sizes = set()
    shapes = set()
    # Iterate through all images in the directory
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        with Image.open(img_path) as img:
            unique_sizes.add(img.size)
            shapes.add(np.array(img).shape)
            print(img.size)
            print(np.array(img).shape)
    return unique_sizes, shapes

#img_dir = 'data/ODIR-5K_Training_Dataset'
#sizes = get_unique_image_sizes(img_dir)