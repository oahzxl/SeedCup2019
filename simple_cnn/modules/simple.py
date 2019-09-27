from torch.nn import Module
from torch import nn
import torch.nn.functional as f
from modules.cnn import CNNCell
from modules.restnet import ResNet
from torchvision import models


class Simple(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Simple, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        # self.cnn = CNNCell()   # with relu and bn
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
