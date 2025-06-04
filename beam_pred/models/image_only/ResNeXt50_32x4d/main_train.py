from src.dataset_loader import BeamPredictionDataset
from models.image_only.ResNeXt50_32x4d.model import get_model
from src.train import train_model

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.optim as optim

# Dönüşümler
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Veri Seti
dataset = BeamPredictionDataset(
    csv_path="../../../dataset/scenario5_dev_train.csv",
    root_dir="../../../dataset/unit1",
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model
model = get_model(num_classes=64)

#  Optimizer & Cihaz
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Eğitimi başlat
train_model(model, dataloader, optimizer, device, num_epochs=5)

# Eğitilen modeli kaydet
torch.save(model.state_dict(), "resnet50_32x4d_beam_model.pth")
print("Model başarıyla kaydedildi: resnet50_32x4d_beam_model.pth")
