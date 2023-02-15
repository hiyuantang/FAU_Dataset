import torch
import numpy as np
import PIL.Image as Image
import cv2
from torchvision import datasets, models, transforms

image_path = '/Volumes/Yuan-T7/Datasets/FAU/images/European_Man/em1/7em1_7.10.png'

data_transforms = {
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
    }


input_image = Image.open(image_path)
input_image = input_image.convert('RGB')
input_image = torch.tensor(np.array(input_image))
print(input_image.shape)
input_image = torch.transpose(input_image, 0, 2).transpose(1, 2)
print(input_image.shape)
input_image = input_image[:, 200:1000, 850:1650]
print(input_image.shape)
input_image = data_transforms['test'](input_image)


print(input_image.shape)

