import torch.nn as nn
from torchvision.models import resnext101_32x8d

def get_model(num_classes=64):
    model = resnext101_32x8d(weights=None)  # pretrained yerine weights=None kullandık
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
