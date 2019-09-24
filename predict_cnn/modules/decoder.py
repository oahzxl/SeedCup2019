import torch
from torch.nn import Module
from torch import nn
from modules.cnn import CNNCell


class Decoder(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.cnn = CNNCell()
        self.to_emb = TimeToEmbed(embedding_dim)

        self.fc_no_label = nn.Linear(in_features=2240, out_features=300)
        self.fc_label = nn.Linear(in_features=2240, out_features=300)

        self.fc_t = nn.Linear(in_features=300, out_features=1)
        self.dropout = nn.Dropout(p=0.3)
        self.double()

    def forward(self, inputs, mode, field):
        if mode == 'train':

            inputs = torch.cat((self.embedding(inputs[:-3]).unsqueeze(1),
                               self.to_emb(inputs[-3:]).unsqueeze(1)), dim=3)

            inputs_no_label = self.cnn(inputs[:10].view(inputs.size(0), inputs.size(1), -1, 60))
            inputs_no_label = self.fc_no_label(inputs_no_label.view(inputs_no_label.size(0), -1))

            inputs_label = self.cnn(inputs.view(inputs.size(0), inputs.size(1), -1, 60))
            inputs_label = self.fc_label(inputs_label.view(inputs_label.size(0), -1))

            inputs = (inputs_label + inputs_no_label) / 2
            inputs = self.fc_t(self.dropout(inputs))

            return inputs, inputs_no_label, inputs_label

        else:
            inputs = self.embedding(inputs).unsqueeze(1)

            inputs = self.cnn(inputs.view(inputs.size(0), inputs.size(1), -1, 60))
            inputs = inputs.view(inputs.size(0), -1)
            inputs = self.fc_no_label(self.dropout(inputs))
            inputs = self.fc_t(self.dropout(inputs))

        return inputs


class TimeToEmbed(Module):
    def __init__(self, embedding_dim):
        super(TimeToEmbed, self).__init__()
        self.fc = nn.Linear(in_features=1, out_features=embedding_dim)

    def forward(self, inputs):
        inputs = (inputs - 50) / 200
        inputs = self.fc(inputs)
        return inputs