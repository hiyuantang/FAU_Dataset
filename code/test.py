import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        layer_activations = [x]  # initialize list to store layer activations
        x = self.fc3(x)
        layer_activations.append(x)
        return x, layer_activations

# Create an instance of the neural network
net = Net()
net.train()
# Generate a random input image and compute the output probabilities
input_image = torch.randn(1, 3, 32, 32)
input_image.requires_grad = True
output, layer_activations = net(input_image)

# Calculate the derivative of the output with respect to the features
target_class = 3  # index of the target class
loss = F.cross_entropy(output, torch.tensor([target_class]))
net.zero_grad()
loss.backward()
feature_derivative = input_image.grad

print(feature_derivative.shape)