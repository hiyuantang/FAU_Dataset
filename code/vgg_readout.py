import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Readout(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(4096, 2622)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = self.dropout6(x)
        x = F.relu(self.fc7(x))
        x = self.dropout7(x)
        return self.fc8(x)