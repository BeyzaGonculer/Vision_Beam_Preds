import torch.nn as nn
from torchvision.models import mobilenet_v2

def get_model(num_classes=64):
    model = mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model
