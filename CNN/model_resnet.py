import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
import torchvision.models as models
import torch

def one_hot_encoded_target(target, num_classes):

    target_size = target.data.cpu().numpy().size
    target = target.data.cpu().numpy()
    one_hot = np.zeros((target_size, num_classes))
    one_hot[np.arange(target_size), target] = 1
    
    if torch.cuda.is_available():
        return torch.from_numpy(one_hot).cuda().float()
    else:
        return torch.from_numpy(one_hot).float()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        
        CE_loss = F.binary_cross_entropy(inputs, targets)#, reduction=None)
        pt = torch.exp(-CE_loss)
        # if pt < 0.3:
        #     self.alpha = 0.8
        # else:

        
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        
        
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class MultiOutputModel(nn.Module):
    def __init__(self, n_age_classes, n_gender_classes, n_ethnicity_classes):
        super().__init__()
        self.n_age_classes = n_age_classes
        self.n_gender_classes = n_gender_classes
        self.n_ethnicity_classes = n_ethnicity_classes

        self.focal_loss = FocalLoss()
        #### FOR RESNET50 ### 
        self.base_model = models.resnet50(pretrained=True)
        self.base_model = torch.nn.Sequential(*(list(self.base_model.children())[:-1]))
        
        # create separate classifiers for our outputs
        self.age = nn.Sequential( 
            nn.Linear(in_features=2048, out_features=n_age_classes)
        )
        self.gender = nn.Sequential( 
            nn.Linear(in_features=2048, out_features=n_gender_classes)
        )
        self.ethnicity = nn.Sequential(
            nn.Linear(in_features=2048, out_features=n_ethnicity_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, 1)
        # print("---- ", x.shape)
        return {
            'age': self.age(x),
            'gender': self.gender(x),
            'ethnicity': self.ethnicity(x)
        }

    def get_loss(self, net_output, ground_truth):
        onehot_age = one_hot_encoded_target(ground_truth['age_labels'], self.n_age_classes)
        age_loss   = self.focal_loss(F.softmax(net_output['age'],dim=1), onehot_age)

        onehot_gender = one_hot_encoded_target(ground_truth['gender_labels'], self.n_gender_classes)
        gender_loss   = self.focal_loss(F.softmax(net_output['gender'],dim=1), onehot_gender)

        # gender_loss = F.cross_entropy(net_output['gender'], ground_truth['gender_labels'])
        onehot_ethnicity = one_hot_encoded_target(ground_truth['ethnicity_labels'], self.n_ethnicity_classes)
        ethnicity_loss   = self.focal_loss(F.softmax(net_output['ethnicity'],dim=1), onehot_ethnicity)

        # ethnicity_loss = F.cross_entropy(net_output['ethnicity'], ground_truth['ethnicity_labels'])
        loss = age_loss + gender_loss + ethnicity_loss
        # loss = age_loss
        return loss, {'age': age_loss, 'gender': gender_loss, 'ethnicity': ethnicity_loss}
