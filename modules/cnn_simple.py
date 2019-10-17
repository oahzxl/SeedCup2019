import torch.nn.functional as f
from torch import nn
from torch.nn import Module


class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.bn1 = nn.BatchNorm1d(14)
        self.cnn1 = nn.Conv1d(in_channels=14, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.cnn5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=0)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=1, padding=0)

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


class SimpleCNN(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(SimpleCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.cnn = CNN()

        self.fc_1 = nn.Linear(in_features=1792, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=1)
        self.dropout = nn.Dropout(p=0.5)

        self.double()

    def forward(self, inputs, mode, field):
        inputs = self.embedding(inputs)
        inputs = self.cnn(inputs)
        inputs = inputs.reshape(inputs.size(0), -1)
        inputs = self.fc_1(self.dropout(inputs))
        inputs = self.fc_2(f.relu(self.dropout(inputs)))
        return inputs
