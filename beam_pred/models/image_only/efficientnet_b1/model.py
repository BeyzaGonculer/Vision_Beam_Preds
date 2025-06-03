import torchvision.models as models
import torch.nn as nn

def get_model(num_classes=64):
    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
