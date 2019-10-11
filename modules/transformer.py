import torch
import torch.nn.functional as f
from torch import nn
from torch.nn import Module


class Transformer(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=300, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        decoder_layer = nn.TransformerDecoderLayer(d_model=300, nhead=2)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.fc1 = nn.Linear(300, 2048)
        self.fc2 = nn.Linear(2048, 1)

        self.dropout = nn.Dropout(p=0.5)
        self.double()

    def forward(self, inputs):
        src = self.embedding(inputs[:, :15])
        tgt = self.embedding(inputs[:, 15:])
        src = self.transformer_encoder(src)
        out = self.transformer_decoder(tgt, src)

        out = out.view(-1, out.size(2))
        out = f.relu(self.fc1(self.dropout(out)))
        out = self.fc2(self.dropout(out))
        out = out.view(inputs.size(0), -1)
        return out
