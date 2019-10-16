import torch.nn.functional as f
from torch import nn
from torch.nn import Module

from modules.cnn_cell import CNNCell


class RNNSVM(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(RNNSVM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.encoder = nn.LSTM(input_size=128, hidden_size=128, bidirectional=True, batch_first=True,
                               num_layers=2, dropout=0.1)

        self.fc_1 = nn.Linear(in_features=3584, out_features=512)
        self.fc_2 = nn.Linear(in_features=512, out_features=64)
        self.fc_3 = nn.Linear(in_features=64, out_features=1)
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
