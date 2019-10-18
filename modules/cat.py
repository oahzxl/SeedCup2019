import torch.nn.functional as f
from torch import nn
from torch.nn import Module
import torch


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
        return inputs


class RNNSVM(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(RNNSVM, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim,
                               bidirectional=True, batch_first=True,
                               num_layers=2, dropout=0.2)
        self.cnn = CNN()

        self.fc_1_1 = nn.Linear(in_features=5120, out_features=512)
        self.fc_1_2 = nn.Linear(in_features=1152, out_features=512)
        self.fc_2 = nn.Linear(in_features=1024, out_features=256)
        self.fc_3 = nn.Linear(in_features=256, out_features=1)
        self.dropout = nn.Dropout(p=0.5)

        self.double()

    def forward(self, inputs, mode, field):
        inputs = self.embedding(inputs)

        inputs1, (_, _) = self.encoder(inputs)
        inputs1 = inputs1.reshape(inputs1.size(0), -1)
        inputs1 = self.fc_1_1(self.dropout(inputs1))

        inputs2 = self.cnn(inputs)
        inputs2 = inputs2.reshape(inputs2.size(0), -1)
        inputs2 = self.fc_1_2(self.dropout(inputs2))

        inputs = torch.cat((inputs1, inputs2), dim=1)
        inputs = self.fc_2(f.relu(self.dropout(inputs)))
        inputs = self.fc_3(f.relu(self.dropout(inputs)))
        return inputs
