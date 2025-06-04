import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BeamPredictionDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
      
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        image_rel_path = self.data.loc[idx, 'unit1_rgb_1'].replace('./unit1/', '')
        image_path = os.path.join(self.root_dir, image_rel_path)

     
        image = Image.open(image_path).convert("RGB")

       
        if self.transform:
            image = self.transform(image)

       
        label = int(self.data.loc[idx, 'beam_index_1'])

        return image, label
