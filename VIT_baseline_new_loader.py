import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
import os
from torchvision import datasets
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
import timm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import numpy as np
import datetime
import imgaug.augmenters as iaa
import cv2


# Define the augmentation techniques without scaling operations
augmenters = [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Affine(rotate=(-45, 45)),
    iaa.GaussianBlur(sigma=(0, 3.0)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
    iaa.Multiply((0.5, 1.5)),
    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
    iaa.Grayscale(alpha=(0.0, 1.0)),
    iaa.ElasticTransformation(alpha=(0.5, 3.0), sigma=0.25),
    iaa.PiecewiseAffine(scale=(0.01, 0.05)),
]

# Dictionary to store original image paths for duplicate check
image_paths = {}

id_counter = 4785

class ODIRDataset(Dataset):
    def __init__(self, dataframe, img_dir, transforms=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)
    
    def augment_and_update(self, row, name, id_counter, id):
        original_image_name = str(id) + '_' + name + ".jpg"
        original_image_path = os.path.join(self.img_dir, original_image_name)
        original_image = cv2.imread(original_image_path)

        selected_augmenter = np.random.choice(augmenters)
        augmented_image = selected_augmenter.augment_image(original_image)

        new_image_name = str(id_counter) + '_' + name + ".jpg"
        #augmented_image_path = os.path.join('/Users/chi-himng/AI/AI for Medical Imaging/test_generator/', new_image_name)
        #cv2.imwrite(augmented_image_path, augmented_image)

        return new_image_name, augmented_image

    def __getitem__(self, idx):
        #left_img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Left-Fundus'])
        #right_img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Right-Fundus'])

        id_counter = 4785  # Initial ID value
        for index, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe)):
            # Check if the row is a duplicate
            if tuple(row) in image_paths:
                # If duplicate, augment the image and update DataFrame
                id_counter += 1
                new_image_name_left, left_image = self.augment_and_update(row['Left-Fundus'], 'left', id_counter, self.dataframe.at[index, 'ID'])
                new_image_name_right, right_image = self.augment_and_update(row['Right-Fundus'], 'right', id_counter, self.dataframe.at[index, 'ID'])

                
                # Update DataFrame with new ID and image name
                self.dataframe.at[index, 'ID'] = id_counter
                self.dataframe.at[index, 'Left-Fundus'] = new_image_name_left
                self.dataframe.at[index, 'Right-Fundus'] = new_image_name_right
            else:
                # If not a duplicate, update the dictionary with the row data
                image_paths[tuple(row)] = index
                left_img_name = os.path.join(self.img_dir, self.dataframe.iloc[image_paths[tuple(row)]]['Left-Fundus'])
                right_img_name = os.path.join(self.img_dir, self.dataframe.iloc[image_paths[tuple(row)]]['Right-Fundus'])
                left_image = Image.open(left_img_name)
                right_image = Image.open(right_img_name)

        values = self.dataframe.iloc[idx][5:].values.astype(np.float32)
        labels = torch.tensor(values)

        return (left_image, right_image), labels

transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),])

# put annotations in current directory
df = pd.read_csv('/Users/chi-himng/AI/AI for Medical Imaging/balanced_df_copy.csv')
df = df.drop(columns=['Left-Diagnostic Keywords', 'Right-Diagnostic Keywords'])


dataset = ODIRDataset(df, '/Users/chi-himng/AI/AI for Medical Imaging/test_generator/', transforms=transform)

train_df, validation_df = train_test_split(df, test_size=0.10, random_state=42)

# Split the dataset into training and validation sets
train_size = int(0.90 * len(dataset))
validation_size = len(dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])


train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=False)

print(train_dataloader.dataset.__len__())


class VisionTransformer(nn.Module):
    def __init__(self, num_classes):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes

        # Vision Transformer backbone
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Replace the final classification layer
        self.backbone.head = nn.Identity()

        # Add a classification head for multi-label classification
        self.classification_head = nn.Linear(self.backbone.embed_dim * 2, num_classes)


    def forward(self, image_left, image_right):
        features_left = self.backbone(image_left)
        features_right = self.backbone(image_right)
        combined_features = torch.cat((features_left, features_right), dim=1)

        # Pass through the classification head
        logits = self.classification_head(combined_features)

        return logits

# Define training and evaluation functions
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader):
        (images_left, images_right), labels = batch
        images_left, images_right, labels = images_left.to(device), images_right.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images_left, images_right)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


#calculate kappa, F-1 socre and AUC value
def ODIR_Metrics(gt_data, pr_data):
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

    kappa, f1, auc, final_score = ODIR_Metrics(all_labels, all_preds)

    return val_loss / len(dataloader), final_score, kappa, f1, auc


def check_trainable_parameters(model):
    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        else:
            frozen_params.append((name, param.numel()))

    #print("Trainable Parameters:")
    #for name, numel in trainable_params:
    #    print(f"{name}: {numel} parameters")

    #print("\nFrozen Parameters:")
    #for name, numel in frozen_params:
    #    print(f"{name}: {numel} parameters")


# Hyperparameters
learning_rate = 0.001
num_epochs = 1
num_classes = 8  # Number of diseases

# Initialize the model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(num_classes).to(device)
model.backbone.requires_grad_(False)
check_trainable_parameters(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for multi-label classification

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
model_checkpoint_name = f'best_model_{timestamp}.pth'

best_score = -1
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
for epoch in range(num_epochs):
    model.train()
    train_loss = train(model, train_dataloader, criterion, optimizer, device)
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