import torch.nn.functional as f
from torch import nn
from torch.nn import Module


class CNNCell(Module):
    def __init__(self):
        super(CNNCell, self).__init__()
        self.bn1 = nn.BatchNorm2d(1)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=2, padding=1)
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1)
        self.cnn4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cnn5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=0)
        self.cnn6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=0)

    def forward(self, inputs):
        inputs = self.bn1(inputs)
        inputs = self.cnn1(inputs)
        inputs = self.cnn2(inputs)
        inputs = f.relu(inputs)
        inputs = self.bn2(inputs)
        inputs = self.cnn3(inputs)
        inputs = self.cnn4(inputs)
        inputs = f.relu(inputs)
        inputs = self.bn3(inputs)
        inputs = self.cnn5(inputs)
        inputs = self.cnn6(inputs)
        inputs = f.relu(inputs)
        return inputs
