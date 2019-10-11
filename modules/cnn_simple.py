import torch.nn.functional as f
from torch import nn
from torch.nn import Module

from modules.restnet import ResNet


class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.bn1 = nn.BatchNorm2d(1)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=2, padding=0)
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.cnn3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=0)
        self.cnn4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=0)

    def forward(self, inputs):
        inputs = self.bn1(inputs)
        inputs = self.cnn1(inputs)
        inputs = self.cnn2(inputs)
        inputs = f.relu(inputs)
        inputs = self.bn2(inputs)
        inputs = self.cnn3(inputs)
        inputs = self.cnn4(inputs)
        inputs = f.relu(inputs)
        return inputs


class SimpleCNN(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(SimpleCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        # self.cnn = CNN()   # with relu and bn
        self.cnn = ResNet(layers=[2, 2, 2, 2])

        self.fc_t1 = nn.Linear(in_features=6144, out_features=512)
        self.fc_t_day = nn.Linear(in_features=512, out_features=1)
        self.fc_t_hour = nn.Linear(in_features=512, out_features=1)
        self.dropout = nn.Dropout(p=0.5)
        self.double()

    def forward(self, inputs, mode, field):
        inputs = self.embedding(inputs).unsqueeze(1)

        inputs = self.cnn(inputs.view(inputs.size(0), inputs.size(1), -1, 60))
        inputs = inputs.view(inputs.size(0), -1)
        inputs = self.fc_t1(inputs)
        day = self.fc_t_day(f.relu(self.dropout(inputs)))
        hour = self.fc_t_hour(f.relu(self.dropout(inputs)))

        return day, hour
