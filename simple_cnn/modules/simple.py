from torch.nn import Module
from torch import nn
import torch.nn.functional as f
from modules.cnn import CNNCell
from torchvision import models


class Simple(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Simple, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        # self.cnn = CNNCell()   # with relu and bn
        self.init_cnn = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.cnn = models.resnet34(pretrained=False)

        self.fc_t1 = nn.Linear(in_features=1000, out_features=512)
        self.fc_t_day = nn.Linear(in_features=512, out_features=1)
        self.fc_t_hour = nn.Linear(in_features=512, out_features=1)
        self.dropout = nn.Dropout(p=0.5)
        self.double()

    def forward(self, inputs, mode, field):
        inputs = self.embedding(inputs).unsqueeze(1)

        inputs = self.init_cnn(inputs.view(inputs.size(0), inputs.size(1), -1, 60))
        inputs = self.cnn(inputs)
        inputs = inputs.view(inputs.size(0), -1)
        inputs = self.fc_t1(inputs)
        day = self.fc_t_day(f.relu(self.dropout(inputs)))
        hour = self.fc_t_hour(f.relu(self.dropout(inputs)))

        return day, hour
