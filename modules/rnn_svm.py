import torch.nn.functional as f
from torch import nn
from torch.nn import Module
import torch


class RNNSVM(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(RNNSVM, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.encoder = nn.LSTM(input_size=128, hidden_size=128, bidirectional=True, batch_first=True,
                               num_layers=2, dropout=0.1)

        self.fc_1 = nn.Linear(in_features=3584, out_features=512)
        self.fc_2 = nn.Linear(in_features=512, out_features=128)
        self.fc_3 = nn.Linear(in_features=128, out_features=1)
        self.dropout = nn.Dropout(p=0.5)

        self.double()

    def forward(self, inputs, mode, field):
        inputs = self.embedding(inputs)
        inputs, (_, _) = self.encoder(inputs)
        inputs = inputs.reshape(inputs.size(0), -1)
        inputs = self.fc_1(self.dropout(inputs))
        inputs = self.fc_2(f.relu(self.dropout(inputs)))
        inputs = self.fc_3(f.relu(self.dropout(inputs)))
        return inputs

    def noise(self, inputs):
        for i in range(inputs.size(0)):
            if torch.rand(size=(1, 1)) > 0.7:
                value = torch.randint(low=0, high=self.num_embeddings, size=(1, 1))
                row = torch.randint(low=0, high=inputs.size(1), size=(1, 1))
                inputs[i, int(row)] = value.squeeze(1)
        return inputs
