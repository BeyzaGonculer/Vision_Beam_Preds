from src.dataset_loader import BeamPredictionDataset
from models.image_only.resnet50.model import get_model
from src.train import train_model

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.optim as optim

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = BeamPredictionDataset(
    csv_path="../../../dataset/scenario5_dev_train.csv",
    root_dir="../../../dataset/unit1",
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = get_model(num_classes=64)

optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_model(model, dataloader, optimizer, device, num_epochs=5)

torch.save(model.state_dict(), "resnet50_beam_model.pth")
print("Model başarıyla kaydedildi: resnet50_beam_model.pth")
