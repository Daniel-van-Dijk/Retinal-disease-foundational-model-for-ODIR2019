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
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
import datetime
from src.models.foundational_model.util.datasets import *
from src.models.foundational_model.util.asymmetric_loss import *
from collections import namedtuple

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
"2957_right.jpg",
"2340_lef.jpg",
"1706_left.jpg",
"1710_right.jpg",
"4522_left.jpg",
"1222_right.jpg", 
"1260_left.jpg", 
"2133_right.jpg", 
"240_left.jpg",
"240_right.jpg",
"150_left.jpg", 
"150_right.jpg",
]
# Manual found low quality: 2340 left, 1706_left, 1710_right, 4522_left, 1222_right, 1260_left
# 2133_right, 240_left, 240_right, 150_left, 150_right



    
disease_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
original_df = pd.read_excel('/home/scur0556/ODIR2019/data/ODIR-5K_Training_Annotations(Updated)_V2.xlsx')
original_df = original_df.drop(columns=['Left-Diagnostic Keywords', 'Right-Diagnostic Keywords'])

low_quality_files_set = set(low_quality_files)


original_df = original_df[~original_df['Left-Fundus'].isin(low_quality_files_set) & ~original_df['Right-Fundus'].isin(low_quality_files_set)]


train_df, validation_df = train_test_split(original_df, test_size=0.2, random_state=42)


Args = namedtuple('Args', ['input_size'])
args = Args(input_size=224)

dataset_train = ODIRDataset(train_df, '/home/scur0556/ODIR2019/data/cropped_ODIR-5K_Training_Dataset', is_train=True, args=args)
dataset_val = ODIRDataset(validation_df, '/home/scur0556/ODIR2019/data/cropped_ODIR-5K_Training_Dataset', is_train=False, args=args)


train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=False)

first_batch = next(iter(train_dataloader))

# Checking the type and length of each element in the batch
print(type(first_batch[0]), len(first_batch[0]))
print(type(first_batch[1]), len(first_batch[1]))


#calculate kappa, F1-score and AUC value
def ODIR_Metrics(gt_data, pr_data):
    """ function from ODIR2019 challenge """
    th = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
    kappa = metrics.cohen_kappa_score(gt, pr>th)
    f1 = metrics.f1_score(gt, pr>th, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa+f1+auc)/3.0
    return kappa, f1, auc, final_score

def evaluate(model, dataloader, device, criterion):
    model.eval()
    all_labels = []
    all_logits = []
    val_loss = 0
    with torch.no_grad():
        for (images_left, images_right), labels in dataloader:
            images_left, images_right = images_left.to(device), images_right.to(device)
            labels = labels.to(device)

            logits = model(images_left, images_right)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_logits = np.vstack(all_logits)

    all_preds = (all_logits > 0.5).astype(np.float32)
    # val_loss, average_score, kappa, f1, auc, val_confusion, val_accuracy
    kappa, f1, auc, final_score = ODIR_Metrics(all_labels, all_preds)

    return val_loss / len(dataloader), final_score, kappa, f1, auc

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
print(model)
#for param in model.resnet.parameters():
#   param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")
criterion=AsymmetricLoss(gamma_neg=2, gamma_pos=2, clip=0)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
model_checkpoint_name = f'best_model_{timestamp}.pth'

def train(model, num_epochs, train_dataloader, validation_dataloader, criterion):

    best_score = -1
    num = 0
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    for epoch in range(num_epochs):

        model.train()  # set the model to training mode
        #pbar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100)
        for batch in train_dataloader:
            #print(batch.shape)
            (images_left, images_right), labels = batch
            optimizer.zero_grad()
            img_left, img_right = images_left.to(device), images_right.to(device)
            labels = labels.to(device)
            outputs = model(img_left, img_right)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # average_metric, kappa, f1, auc
        val_loss, average_score, kappa, f1, auc = evaluate(model, validation_dataloader, device, criterion)
        scheduler.step(val_loss)
        if average_score > best_score:
            best_score = average_score
            torch.save(model.state_dict(), model_checkpoint_name)
            print(f"Score increased to {average_score:.4f}. Model saved!")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  average score: {average_score:.4f}")
        print(f"  f1: {f1:.4f}")
        print(f"  kappa: {kappa:.4f}")
        print(f"  auc: {auc:.4f}")
        #print(f"  Val Accuracy: {val_accuracy:.4f}")
        #print("  Val Confusion Matrix:")
        #print(val_confusion)

train(model, 50, train_dataloader, validation_dataloader, criterion)

