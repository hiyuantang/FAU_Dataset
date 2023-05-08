import os 
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import cv2
import json
import numpy as np

class facegenDataset(Dataset):
    def __init__(self, root_dir, subjects, mode, transform=None):
        self.root_dir = root_dir
        self.images_dir = os.path.join(self.root_dir, 'images')
        self.labels_dir = os.path.join(self.root_dir, 'labels')
        self.summary_dir = os.path.join(self.labels_dir, 'summary.txt')
        self.subjects = np.array(subjects)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        count = 0
        for i in next(os.walk(self.images_dir))[1]:
            updated_path = os.path.join(self.images_dir, i)
            for j in next(os.walk(updated_path))[1]:
                if j in self.subjects:
                    target_path = os.path.join(updated_path, j)
                    for _, _, files in os.walk(target_path):
                        for z in files:
                            if z.startswith('._'):
                                files.remove(z)
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
                    if ('\\'+i+'\\') in j or ('/'+i+'/') in j:
                        target_item_dict[count] = j
                        count += 1
        label_path = target_item_dict[index]
        image_path = label_path.replace('labels', 'images')
        image_path = image_path.replace('.txt', '.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #######################
        # image size: (1080, 1920, 3) --> (800, 800, 3)
        image = torch.tensor(image[100:700, 50:650, 0:3])

        if self.mode == 0:
            image = image.permute(2,0,1)
        else:
            img_around = F.pad(image, (0,0,1,1,1,1), 'constant', 0)
            #img_pos0 = F.pad(image, (0,0,2,0,2,0), 'constant', 0)
            img_pos1 = F.pad(image, (0,0,0,2,0,2), 'constant', 0)
            #img_pos2 = F.pad(image, (0,0,0,2,2,0), 'constant', 0)
            #img_pos3 = F.pad(image, (0,0,2,0,0,2), 'constant', 0)
            if self.mode == 1:
                image = img_around - img_pos1
            elif self.mode == 2:
                image = img_around - img_pos1
                image = (image - image.min())/(image.max()-image.min())
                image_re = 1.0 / (image + 1.0)
                image = image + image_re
            elif self.mode == 3:
                image = img_around - img_pos1
                image = (image - image.min())/(image.max()-image.min())
                image = np.arctan(image - 0.5)
                image = np.abs(image)
            elif self.mode == 4:
                kernel = np.array([[0, -1, 0],
                                    [-1, 5,-1],
                                    [0, -1, 0]])
                image_sharp = cv2.filter2D(src=image.numpy(), ddepth=-1, kernel=kernel)
                image_sharp = torch.tensor(image_sharp)
                image_sharp = (image_sharp - image_sharp.min())/(image_sharp.max()-image_sharp.min())
                img_around = F.pad(image_sharp, (0,0,1,1,1,1), 'constant', 0)
                img_pos0 = F.pad(image_sharp, (0,0,0,2,2,0), 'constant', 0)
                image = img_around - img_pos0
            else:
                print('mode error')
            image = (image - image.min())/(image.max()-image.min())
            image = image.permute(2,0,1)
            #######################

        with open(label_path) as f:
            data = json.load(f)
            labels = torch.tensor(list(data.values()))

        if self.transform:
            image = self.transform(image)
        
        return (image, labels)

