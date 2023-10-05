
import torch
import torch.nn as nn
import models_vit
from models_vit import VisionTransformer
from timm.models.layers import trunc_normal_

class ODIRmodel(nn.Module):
    def __init__(self, base_vit_model: VisionTransformer, num_classes: int):
        super(ODIRmodel, self).__init__()
        self.base_vit_model = base_vit_model
        self.classifier = nn.Linear(2048, num_classes)  # Concatenate two 1024-dim features
        self.base_vit_model.head = nn.Identity()
        trunc_normal_(self.classifier.weight, std=2e-5)

    def forward(self, left_image, right_image):
        left_features = self.base_vit_model(left_image)
        right_features = self.base_vit_model(right_image)
        
        combined_features = torch.cat([left_features, right_features], dim=-1)
        
        output = self.classifier(combined_features)
        return output