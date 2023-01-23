import os 
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2

class FAUDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img_path = os.path.joint(self.root_dir, self.labels)
        image = cv2.imraed(img_path)
        y_label = torch.tensor(int(self.labels.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)

