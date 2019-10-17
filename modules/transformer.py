import torch
import torch.nn.functional as f
from torch import nn
from torch.nn import Module


class FinalFC(Module):
    def __init__(self, d_model):
        super(FinalFC, self).__init__()
        self.fc_1 = nn.Linear(d_model, 2048)
        self.fc_2 = nn.Linear(2048, 512)
        self.fc_3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.double()

    def forward(self, inputs):
        out = f.relu(self.fc_1(self.dropout(inputs)))
        out = f.relu(self.fc_2(self.dropout(out)))
        out = self.fc_3(self.dropout(out))
        return out


class Transformer(Module):
    def __init__(self, num_embeddings, embedding_dim, d_model, nhead, num_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.fc = FinalFC(12 * self.d_model)

        self.double()

    def forward(self, inputs, field, train=True):
        inputs = self.embedding(inputs).permute(1, 0, 2)
        inputs = self.transformer_encoder(inputs).permute(1, 0, 2)
        inputs = self.fc(inputs.reshape(inputs.size(0), -1))
        return inputs

    @staticmethod
    def make_mask():

        tgt_mask = torch.zeros((4, 4), dtype=torch.double)

        for i in range(1, 4):
            for j in range(i):
                tgt_mask[i, j] = float('-inf')

        return tgt_mask
