
import torch
import torch.nn as nn
import models_vit
from models_vit import VisionTransformer
from timm.models.layers import trunc_normal_

class ODIRmodel(nn.Module):
    def __init__(self, base_vit_model: VisionTransformer, num_classes: int):
        super(ODIRmodel, self).__init__()
        self.base_vit_model = base_vit_model
        self.base_vit_model.head = torch.nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 8))
        #self.base_vit_model.head = nn.Identity()
        #trunc_normal_(self.classifier.weight, std=2e-5)
        trunc_normal_(self.base_vit_model.head[0].weight, std=2e-5)
        trunc_normal_(self.base_vit_model.head[3].weight, std=2e-5)
        #trunc_normal_(self.base_vit_model.head[6].weight, std=2e-5)
        #trunc_normal_(self.base_vit_model.head[6].weight, std=2e-5)
        #self.base_vit_model.head = nn.Linear(1024, 8)
        #trunc_normal_(self.base_vit_model.head.weight, std=2e-5)

    def forward(self, image):
        #left_features = self.base_vit_model.forward_features(left_image)
        features = self.base_vit_model.forward_features(image)
        
        #combined_features = torch.cat([left_features, right_features], dim=-1)
        
        output = self.base_vit_model.head(features)
        return output
    
    def no_weight_decay(self):
        no_decay = {'bias', 'LayerNorm.weight'}
        return [param for name, param in self.named_parameters() if not any(nd in name for nd in no_decay)]
