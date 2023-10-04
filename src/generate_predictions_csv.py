import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

from models.ViT_model import VisionTransformer


from torchvision import datasets
from tqdm.notebook import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torchvision.models import ResNet50_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
import torch.nn.functional as F



class TestDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),])

test_loader = DataLoader(TestDataset('/home/scur0547/ODIR2019/data/cropped_ODIR-5K_Testing_Images', transform=transform), batch_size=64, shuffle=False)



def save_predictions(model, dataloader, device, output_file, logit_output=True):

    model.eval()
    all_ids = []
    all_probs = []



    with torch.no_grad():
        for (images_left, images_right, image_ids) in tqdm(dataloader):
            images_left, images_right = images_left.to(device), images_right.to(device)

            output = model(images_left, images_right)
            # set logit_output = True when model outputs logits so NO sigmoid applied in forward of model
            # set logit_output = False when model outputs probs
            if logit_output:
              output = F.sigmoid(output)  # Convert logits to probabilities

            all_ids.extend(image_ids.cpu().numpy())
            all_probs.append(output.cpu().numpy())

    all_probs = np.vstack(all_probs)

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])

        for idx, prob in zip(all_ids, all_probs):
            writer.writerow([idx] + list(prob))


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)



model = VisionTransformer(8).to(device)
checkpoint = torch.load("/home/scur0547/ODIR2019/best_model_20231003_1437.pth", map_location=device)
model.load_state_dict(checkpoint)

# check logit_output param
save_predictions(model, test_loader, 'cuda', 'prob_predictions.csv', logit_output=True)
save_predictions(model, test_loader, 'cuda', 'logit_predictions.csv', logit_output=False)