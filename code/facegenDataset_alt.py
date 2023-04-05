import os 
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import cv2
import json
import numpy as np

class facegenDataset(Dataset):
    def __init__(self, root_dir, subjects, transform=None):
        self.root_dir = root_dir
        self.images_dir = os.path.join(self.root_dir, 'images')
        self.labels_dir = os.path.join(self.root_dir, 'labels')
        self.summary_dir = os.path.join(self.labels_dir, 'summary.txt')
        self.subjects = np.array(subjects)
        self.transform = transform

    def __len__(self):
        count = 0
        for i in next(os.walk(self.images_dir))[1]:
            updated_path = os.path.join(self.images_dir, i)
            for j in next(os.walk(updated_path))[1]:
                if j in self.subjects:
                    target_path = os.path.join(updated_path, j)
                    for _, _, files in os.walk(target_path):
                        count += len(files)
        return count
    
    def __getitem__(self, index):
        target_item_dict = {}
        with open(self.summary_dir) as f:
            data = json.load(f)
            data = list(data.values())
            count = 0
            for i in self.subjects:
                for j in data:
                    if ('\\'+i+'\\') in j:
                        target_item_dict[count] = j
                        count += 1
        label_path = target_item_dict[index]
        image_path = label_path.replace('labels', 'images')
        image_path = image_path.replace('.txt', '.png')
        image = cv2.imread(image_path)
        #######################
        # image size: (1080, 1920, 3) --> (800, 800, 3)
        image = torch.tensor(image[100:700, 50:650, 0:3])
        img_around = F.pad(image, (0,0,1,1,1,1), 'constant', 0)
        img_pos0 = F.pad(image, (0,0,2,0,2,0), 'constant', 0)
        image = (img_around - img_pos0 + 1.0)/2.0
        image = image.permute(2,0,1)
        #######################

        with open(label_path) as f:
            data = json.load(f)
            labels = torch.tensor(list(data.values()))

        if self.transform:
            image = self.transform(image)
        
        return (image, labels)

