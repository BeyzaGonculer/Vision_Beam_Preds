import torch.nn as nn
from torchvision import models

def get_model(num_classes=64):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
