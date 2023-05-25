# -*- coding: utf-8 -*- 
# got from https://github.com/prlz77/vgg-face.pytorch
# __author__ = "Pau Rodr�guez L�pez, ISELAB, CVC-UAB"
# __email__ = "pau.rodri1@gmail.com"

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile


class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super(VGG_16, self).__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(4096, 2622)

    def load_weights(self, path="./../../VGG_FACE.t7"):
        """ Function to load luatorch weights
        Args:
            path: path for the luatorch weights
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if hasattr(layer, "weight") and layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.from_numpy(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.from_numpy(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.from_numpy(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.from_numpy(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward
        Args:
            x: input image (224x224)
        Returns: class logits
        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = self.dropout6(x)
        x = F.relu(self.fc7(x))
        x = self.dropout7(x)
        return self.fc8(x)

    
    def forward_thoughts(self, x, cut):
        x_0 = F.relu(self.conv_1_1(x))
        x_0 = F.relu(self.conv_1_2(x_0))
        x_0 = F.max_pool2d(x_0, 2, 2)

        x_1 = F.relu(self.conv_2_1(x_0))
        x_1 = F.relu(self.conv_2_2(x_1))
        x_1 = F.max_pool2d(x_1, 2, 2)

        x_2 = F.relu(self.conv_3_1(x_1))
        x_2 = F.relu(self.conv_3_2(x_2))
        x_2 = F.relu(self.conv_3_3(x_2))
        x_2 = F.max_pool2d(x_2, 2, 2)

        x_3 = F.relu(self.conv_4_1(x_2))
        x_3 = F.relu(self.conv_4_2(x_3))
        x_3 = F.relu(self.conv_4_3(x_3))
        x_3 = F.max_pool2d(x_3, 2, 2)

        x_4 = F.relu(self.conv_5_1(x_3))
        x_4 = F.relu(self.conv_5_2(x_4))
        x_4 = F.relu(self.conv_5_3(x_4))
        x_4 = F.max_pool2d(x_4, 2, 2)

        x_5 = x_4.view(x_4.size(0), -1)
        x_5 = F.relu(self.fc6(x_5))
        x_5 = self.dropout6(x_5)
        x_5 = F.relu(self.fc7(x_5))
        x_5 = self.dropout7(x_5)
        x_5 = self.fc8(x_5)
        
        if cut == 0:
            return x_0
        elif cut == 1:
            return x_1
        elif cut == 2:
            return x_2
        elif cut == 3:
            return x_3
        elif cut == 4:
            return x_4
        elif cut == 5:
            return x_5
    
    def forward_0(self, x):
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = self.dropout6(x)
        x = F.relu(self.fc7(x))
        x = self.dropout7(x)
        return self.fc8(x)

    def forward_1(self, x):
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = self.dropout6(x)
        x = F.relu(self.fc7(x))
        x = self.dropout7(x)
        return self.fc8(x)

    def forward_2(self, x):
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = self.dropout6(x)
        x = F.relu(self.fc7(x))
        x = self.dropout7(x)
        return self.fc8(x)
    
    def forward_3(self, x):
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = self.dropout6(x)
        x = F.relu(self.fc7(x))
        x = self.dropout7(x)
        return self.fc8(x)
    
    def forward_4(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = self.dropout6(x)
        x = F.relu(self.fc7(x))
        x = self.dropout7(x)
        return self.fc8(x)

if __name__ == "__main__":
    model = VGG_16()
    model.load_weights()
    im = cv2.imread("images/ak.png")
    im = torch.Tensor(im).permute(2, 0, 1).view(1, 3, 224, 224)
    import numpy as np

    im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1)
    preds = F.softmax(model(im), -1)
    values, indices = preds.max(-1)
    print(indices)