import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=64):
    """
    Beam tahmini için ResNet18 modelini yükler.
    Son katmanı beam index sayısına göre ayarlar.
    """
    model = models.resnet18(pretrained=True)

    # Son katmanı değiştir (fc → full-connected layer)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
