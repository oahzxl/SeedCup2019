import torch
import torch.nn.functional as f
from torch import nn
from torch.nn import Module


class Transformer(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=300, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=300, nhead=1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.double()

    def forward(self, inputs):
        src = self.embedding(inputs[:, 15])
        tgt = self.embedding(inputs[15:])
        src = self.transformer_encoder(src)
        out = self.transformer_decoder(tgt, src)
        return out


a = Transformer()
