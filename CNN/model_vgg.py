import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
import torchvision.models as models
import torch
from model_resnet import one_hot_encoded_target
from model_resnet import FocalLoss

class MultiOutputModel(nn.Module):
    def __init__(self, n_age_classes, n_gender_classes, n_ethnicity_classes):
        super().__init__()
        self.n_age_classes = n_age_classes
        self.n_gender_classes = n_gender_classes
        self.n_ethnicity_classes = n_ethnicity_classes

        #### FOR VGG19 
        self.base_model = models.vgg19(pretrained=True).features
        self.pool = nn.AdaptiveAvgPool2d(output_size=(7,7))
        self.focal_loss = FocalLoss()


        self.age = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=n_age_classes, bias=True)
        )
        self.gender = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=n_gender_classes, bias=True)
        )
        self.ethnicity = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=n_ethnicity_classes, bias=True)
        )
    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return {
            'age': self.age(x),
            'gender': self.gender(x),
            'ethnicity': self.ethnicity(x)
        }

        #### FOR VGG19

    def get_loss(self, net_output, ground_truth):
        onehot_age = one_hot_encoded_target(ground_truth['age_labels'], self.n_age_classes)
        age_loss   = self.focal_loss(F.softmax(net_output['age'],dim=1), onehot_age)

        onehot_gender = one_hot_encoded_target(ground_truth['gender_labels'], self.n_gender_classes)
        gender_loss   = self.focal_loss(F.softmax(net_output['gender'],dim=1), onehot_gender)

        onehot_ethnicity = one_hot_encoded_target(ground_truth['ethnicity_labels'], self.n_ethnicity_classes)
        ethnicity_loss   = self.focal_loss(F.softmax(net_output['ethnicity'],dim=1), onehot_ethnicity)

        loss = age_loss + gender_loss + ethnicity_loss
        return loss, {'age': age_loss, 'gender': gender_loss, 'ethnicity': ethnicity_loss}
 