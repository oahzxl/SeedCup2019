from torch.nn import Module
from torch import nn
import torch
import torch.nn.functional as f
from modules.cnn import CNNCell
from modules.restnet import ResNet
from torchvision import models


class Simple(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Simple, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.cnn = CNNCell()   # with relu and bn
        # self.cnn = ResNet(layers=[2, 2, 2, 2])

        self.fc_1 = nn.Linear(in_features=2560, out_features=600)
        self.fc_t_day = nn.Linear(in_features=600, out_features=1)
        self.fc_t_hour = nn.Linear(in_features=600, out_features=1)
        self.decoder = nn.LSTMCell(input_size=600, hidden_size=600)
        self.dropout = nn.Dropout(p=0.5)
        self.double()

    def forward(self, inputs, mode, field):
        inputs = self.embedding(inputs).unsqueeze(1)

        inputs = self.cnn(inputs.view(inputs.size(0), inputs.size(1), -1, 60))
        inputs = inputs.view(inputs.size(0), -1)
        inputs = self.fc_1(inputs)

        outputs = []
        hx = torch.zeros_like(inputs)
        cx = torch.zeros_like(inputs)
        for i in range(4):
            hx, cx = self.decoder(inputs, (hx, cx))
            day = self.fc_t_day(hx)
            hour = self.fc_t_hour(hx)
            outputs.append(day)
            outputs.append(hour)

            day = self.embedding(self.time_to_idx(day, field, 'd').long()).squeeze(1)
            hour = self.embedding(self.time_to_idx(hour, field, 't').long()).squeeze(1)
            inputs = torch.cat((day, hour), dim=1)

        return outputs

    @staticmethod
    def time_to_idx(time, field, mode):
        if mode == 'd':
            for b in range(time.size(0)):
                time[b] = field.vocab.stoi['%.0f' % (time[b] * 8 + 3) + '_' + mode]
        elif mode == 'h':
            for b in range(time.size(0)):
                time[b] = field.vocab.stoi['%.0f' % (time[b] * 10 + 15) + '_' + mode]
        return time
