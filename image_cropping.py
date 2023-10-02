import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import datetime
import matplotlib.pyplot as plt
import cv2
import time
import shutil


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

        # Compute threshold from the horizontal profile
        Vthreshold = np.max(vertical_profile) * 0.06
        
        # Identify transitions based on the threshold
       
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
        
        # use maximum and minimum transition because more than 2 transitions can be found
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



# put annotations in current directory
df = pd.read_excel('data/ODIR-5K_Training_Annotations(Updated)_V2.xlsx')
df = df.drop(columns=['Left-Diagnostic Keywords', 'Right-Diagnostic Keywords'])


def crop_train_images(df, train_dir, cropped_dir):
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir)
    start_time = time.time()
    
    for idx, row in df.iterrows():
        left_img_name = os.path.join(train_dir, row['Left-Fundus'])
        right_img_name = os.path.join(train_dir, row['Right-Fundus'])

        left_img_check = fov_extraction(left_img_name, row['Left-Fundus'], cropped_dir)
        right_img_check = fov_extraction(right_img_name, row['Right-Fundus'], cropped_dir)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

# crop_train_images(df, data/ODIR-5K_Training_Dataset, data/cropped_ODIR-5K_Training_Dataset)

def crop_test_images(test_dir, cropped_dir):
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir)

    for filename in os.listdir(test_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(test_dir, filename)
            restult = fov_extraction(img_path, filename, cropped_dir)


crop_test_images("data/ODIR-5K_Testing_Images", "data/cropped_ODIR-5K_Testing_Images")
