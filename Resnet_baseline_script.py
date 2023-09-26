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

        left_image = Image.open(left_img_name)
        right_image = Image.open(right_img_name)

        values = self.dataframe.iloc[idx][5:].values.astype(np.float32)
        labels = torch.tensor(values)

        if self.transforms:
            left_image = self.transforms(left_image)
            right_image = self.transforms(right_image)

        return (left_image, right_image), labels

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),])

# put annotations in current directory
df = pd.read_excel('data/ODIR-5K_Training_Annotations(Updated)_V2.xlsx')
df = df.drop(columns=['Left-Diagnostic Keywords', 'Right-Diagnostic Keywords'])


train_df, validation_df = train_test_split(df, test_size=0.10, random_state=42)

train_dataset = ODIRDataset(train_df, 'data/ODIR-5K_Training_Dataset', transforms=transform)
validation_dataset = ODIRDataset(validation_df, 'data/ODIR-5K_Training_Dataset', transforms=transform)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=False)


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
        for (images_left, images_right), labels in tqdm(dataloader):
            images_left, images_right = images_left.to(device), images_right.to(device)
            labels = labels.to(device)

            logits = model(images_left, images_right)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_logits = np.vstack(all_logits)

    all_preds = (all_logits > 0.5).astype(np.float32)
    # val_loss, average_score, kappa, f1, auc, val_confusion, val_accuracy
    kappa, f1, auc, final_score = ODIR_Metrics(all_labels, all_preds)

    return val_loss / len(dataloader), final_score, kappa, f1, auc

class ResNet_baseline(nn.Module):
    def __init__(self, num_diseases=8):
        super(ResNet_baseline, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.fc1 = nn.Linear(2 * 1000, 256)
        self.fc2 = nn.Linear(256, num_diseases)

    def forward(self, img_left, img_right):
        x_left = self.resnet(img_left)
        x_right = self.resnet(img_right)
        x = torch.cat((x_left, x_right), dim =1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        #x = torch.sigmoid(self.fc2(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ResNet_baseline().to(device)
for param in model.resnet.parameters():
   param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")

# from efficientnet paper
loss_weight = torch.tensor([1, 1.2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.2]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
model_checkpoint_name = f'best_model_{timestamp}.pth'

def train(model, num_epochs, train_dataloader, validation_dataloader):

    best_score = -1
    num = 0
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    for epoch in range(num_epochs):

        model.train()  # set the model to training mode
        pbar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100)
        for (img_left, img_right), labels in pbar:
            optimizer.zero_grad()
            img_left, img_right = img_left.to(device), img_right.to(device)
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

train(model, 50, train_dataloader, validation_dataloader)


# class TestDataset(Dataset):
#     def __init__(self, folder_path, transform=None):
#         self.folder_path = folder_path
#         self.transform = transform
#         self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

#     def __len__(self):
#         return len(self.image_files) // 2

#     def __getitem__(self, idx):
#         left_image_name = self.image_files[2 * idx]
#         right_image_name = self.image_files[2 * idx + 1]

#         image_id = int(left_image_name.split('_')[0])

#         left_image = Image.open(os.path.join(self.folder_path, left_image_name))
#         right_image = Image.open(os.path.join(self.folder_path, right_image_name))

#         if self.transform:
#             left_image = self.transform(left_image)
#             right_image = self.transform(right_image)

#         return left_image, right_image, image_id

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),])

# test_loader = DataLoader(TestDataset('ODIR-5K_Testing_Images', transform=transform), batch_size=64, shuffle=False)