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
import torchvision.models as models
import csv
import datetime
from models.foundational_model.util.datasets import *
from models.foundational_model.util.asymmetric_loss import *
from collections import namedtuple
import torch.nn.functional as F


Args = namedtuple('Args', ['input_size'])
args = Args(input_size=224)

test_loader = DataLoader(TestDataset('/home/scur0556/ODIR2019/data/cropped_ODIR-5K_Testing_Images', is_train=False, args=args), batch_size=16, shuffle=False)



def save_predictions(model, dataloader, device, output_file, logit_output=True):

    model.eval()
    all_ids = []
    all_probs = []



    with torch.no_grad():
        for (images_left, images_right, image_ids) in dataloader:
            images_left, images_right = images_left.to(device), images_right.to(device)

            output = model(images_left, images_right)
            # set logit_output = True when model outputs logits so NO sigmoid applied in forward of model
            # set logit_output = False when model outputs probs
            if logit_output:
              output = torch.sigmoid(output)  # Convert logits to probabilities

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

class ResNet_baseline(nn.Module):
    def __init__(self, num_diseases=8):
        super(ResNet_baseline, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.fc1 = nn.Linear(2 * 2048, 8)
        #self.fc2 = nn.Linear(256, num_diseases)

    def forward(self, img_left, img_right):
        x_left = self.resnet(img_left)
        x_right = self.resnet(img_right)
        x = torch.cat((x_left, x_right), dim =1)
        #x = nn.ReLU()(self.fc1(x))
        x = self.fc1(x)
        #x = torch.sigmoid(self.fc2(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ResNet_baseline().to(device)
checkpoint = torch.load("/home/scur0556/ODIR2019/best_model_20231024_1056.pth", map_location=device)
model.load_state_dict(checkpoint)

# check logit_output param
save_predictions(model, test_loader, 'cuda', 'resnet_baseline1.csv', logit_output=True)
#save_predictions(model, test_loader, 'cuda', 'resnet_baseline2.csv', logit_output=False)