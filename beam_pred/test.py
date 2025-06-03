
import torch
from torch.utils.data import DataLoader
from src.beam_multi_dataset import BeamMultiDataset
import pandas as pd

# 🔍 Dosya yolları
CSV_PATH = "dataset/scenario5_dev_train.csv"
ROOT_DIR = "dataset"

# 🧪 Dataset ve DataLoader
dataset = BeamMultiDataset(csv_path=CSV_PATH, root_dir=ROOT_DIR)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 👀 İlk batch'i kontrol et
for batch in loader:
    images, power_values, labels = batch

    print("🔹 Image Tensor Shape:", images.shape)         # [B, 3, 224, 224]
    print("🔸 Power Tensor Shape:", power_values.shape)   # [B, 64]
    print("🎯 Labels:", labels)                           # [B]

    break  # Sadece ilk batch yeterli


df = pd.read_csv("./dataset/scenario5_dev_train.csv")
print(df.head())  # İlk 5 satırı yazdırır
print(df.columns)