# From https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import sys

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    data = datasets.MNIST(
        root="data", train=True, download=True, transform=transforms.Compose([
            transforms.Scale(32),
            transforms.ToTensor()
        ]))
    net = MNISTNet()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for inp, label in data:
        inp = inp.unsqueeze(0) # 1 sample batch
        output = net(inp)

        target = torch.zeros(1, 10)
        target[0, label] = 1

        criterion = nn.MSELoss()
        loss = criterion(output, target)
        
        pred = torch.argmax(output)
        if int(label) == int(pred):
            print('.', end=' ')
        else:
            print('x', end=' ')
        
        sys.stdout.flush()
        
        net.zero_grad()
        loss.backward()
        optimizer.step()
