from models.image_only.resnet50.model import get_model
from src.dataset_loader import BeamPredictionDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch

model = get_model(num_classes=64)
model.load_state_dict(torch.load("resnet50_beam_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = BeamPredictionDataset(
    csv_path="../../../dataset/scenario5_dev_val.csv",  # ✨ validation set buradan
    root_dir="../../../dataset/unit1",
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

top1_correct = 0
top2_correct = 0
top3_correct = 0
total = 0

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        topk = torch.topk(outputs, k=3, dim=1).indices

        top1_correct += (topk[:, 0] == labels).sum().item()

        for i in range(labels.size(0)):
            if labels[i] in topk[i, :2]:
                top2_correct += 1
            if labels[i] in topk[i, :3]:
                top3_correct += 1

        total += labels.size(0)

print(f"Top-1 Accuracy (VAL): {top1_correct / total:.4f} ({100 * top1_correct / total:.2f}%)")
print(f"Top-2 Accuracy (VAL): {top2_correct / total:.4f} ({100 * top2_correct / total:.2f}%)")
print(f"Top-3 Accuracy (VAL): {top3_correct / total:.4f} ({100 * top3_correct / total:.2f}%)")
