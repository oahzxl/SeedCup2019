import torch.nn.functional as f
from torch import nn
from torch.nn import Module


class CNNCell(Module):
    def __init__(self):
        super(CNNCell, self).__init__()
        self.bn1 = nn.BatchNorm1d(512)
        self.cnn1 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=512, out_channels=768, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(768)
        self.cnn3 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=2, stride=1, padding=1)
        self.cnn4 = nn.Conv1d(in_channels=768, out_channels=1024, kernel_size=2, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(1024)
        self.cnn5 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=2, stride=1, padding=0)
        self.cnn6 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=2, stride=1, padding=0)

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
