
import torch
import torch.nn as nn
import models_vit
from models_vit import VisionTransformer
from timm.models.layers import trunc_normal_

class ODIRmodel(nn.Module):
    def __init__(self, base_vit_model: VisionTransformer, num_classes: int):
        super(ODIRmodel, self).__init__()
        self.base_vit_model = base_vit_model
        self.base_vit_model.head = nn.Linear(2048, 8)
    # Concatenate two 1024-dim features
        # self.base_vit_model.head = torch.nn.Sequential(
        # nn.Linear(2048, 512) ,
        # nn.LayerNorm(512),
        # nn.GELU(),
        # nn.Linear(512, 128),
        # nn.LayerNorm(128),
        # nn.GELU(),
        # nn.Linear(128, 8)) 
        ##MIL PART
        # self.L = 128 #self.dim//3
        # self.D = 76 #self.dim//5
        # self.K = 8 #self.num_classes*1
        # self.MIL_Prep = torch.nn.Sequential(
        #         torch.nn.Linear(2048, self.L),
        #         # torch.nn.BatchNorm1d(num_patches),
        #         torch.nn.LayerNorm(self.L),
        #         torch.nn.ReLU(inplace=True),
        #         # nn.Dropout(0.1)
        #         )
        # self.MIL_attention = nn.Sequential(
        #     nn.Linear(self.L, self.D),
        #     # nn.Tanh(),
        #     # nn.BatchNorm1d(num_patches),
        #     torch.nn.LayerNorm(self.D),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(self.D, self.K)

        #     # nn.Linear(self.L, self.K)
        # )

        # self.MIL_classifier = nn.Sequential(
        #     nn.Linear(self.L*self.K, 8),
        # )

        # self.MIL_Prep[0].apply(self._init_weights)
        # self.MIL_Prep[1].apply(self._init_weights)
        # self.MIL_attention[0].apply(self._init_weights)
        # self.MIL_attention[1].apply(self._init_weights)
        # self.MIL_attention[4].apply(self._init_weights)
        # self.MIL_classifier[0].apply(self._init_weights)
      

        # self.base_vit_model.head.apply(self._init_weights)
        trunc_normal_(self.base_vit_model.head.weight, std=2e-5)

    def forward(self, left_image, right_image):
        #Now it returns only features, but want patch information for MIL too
        left_features = self.base_vit_model.forward_features(left_image)
        right_features = self.base_vit_model.forward_features(right_image)
       
        # combined_patch = torch.cat([left_patch, right_patch], dim=-1)

        combined_features = torch.cat([left_features, right_features], dim=-1)
        # print("COMBINED PATCH SHAPE", combined_patch.shape)
        output = self.base_vit_model.head(combined_features)

        ## MIL PART
        # H = self.MIL_Prep(combined_patch)  #B*N*D -->  B*N*L

        # A = self.MIL_attention(H)  # B*N*K
        # # A = torch.transpose(A, 1, 0)  # KxN
        # A = A.permute((0, 2, 1))  #B*K*N
        # A = nn.functional.softmax(A, dim=2)  # softmax over N
        # M = torch.bmm(A, H)  # B*K*N X B*N*L --> B*K*L
        # M = M.view(-1, M.size(1)*M.size(2))

        # mil_out = self.MIL_classifier(M)

        # return vt_out, mil_out
        if self.training:
            return output#, mil_out
        else:
            # during inference, return the average of both classifier predictions
            return output#+ mil_out) / 2
    
    def no_weight_decay(self):
        no_decay = {'bias', 'LayerNorm.weight'}
        return [param for name, param in self.named_parameters() if not any(nd in name for nd in no_decay)]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.00002)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)