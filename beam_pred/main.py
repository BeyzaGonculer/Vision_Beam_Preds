from src.dataset_loader import BeamPredictionDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.image_only.resnet18.model import get_model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


dataset = BeamPredictionDataset(
    csv_path="dataset/scenario5_dev_train.csv",
    root_dir="dataset/unit1",
    transform=transform
)


dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


for images, labels in dataloader:
    print("Görüntü batch boyutu:", images.shape)  # örn: [8, 3, 224, 224]
    print("Label batch:", labels)
    break




model = get_model(num_classes=64)
print(model)
