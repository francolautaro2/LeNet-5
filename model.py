import torch
from torch import nn
import torch.nn.functional as F


# lenet let 32x32 input

class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5) # 6 filters, kernel size 5x5 c1
        self.pool1 = nn.AvgPool2d(2,2) # subsampling layer 2x2 s2
        self.conv2 = nn.Conv2d(6, 16, 5) # 16 filters, kernel size 5x5 c3
        self.pool2 = nn.AvgPool2d(2,2) # subsampling layer 2x2 s4
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # full connected layer c5, 16x5x5 = 400 neurons
        self.fc2 = nn.Linear(120, 84) # full connected layer f6
        self.fc3 = nn.Linear(84, 10) # 10 output classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # C3 -> S4
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # flatten input
        x = x.view(-1, 16 * 5 * 5)
        
        # Fully Connected Layer C5
        x = F.relu(self.fc1(x))
        
        # F6 Fully Connected Layer
        x = F.relu(self.fc2(x))
        
        # output softmax
        x = self.fc3(x)
        return x



