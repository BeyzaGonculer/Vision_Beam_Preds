import torch.nn as nn
from torchvision import models

def get_model(num_classes=64):
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model