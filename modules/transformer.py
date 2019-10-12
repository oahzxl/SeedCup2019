import torch
import torch.nn.functional as f
from torch import nn
from torch.nn import Module


class FinalFC(Module):
    def __init__(self, d_model):
        super(FinalFC, self).__init__()
        self.fc_1 = nn.Linear(d_model, 2048)
        self.fc_2 = nn.Linear(2048, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.double()

    def forward(self, inputs):
        out = f.relu(self.fc_1(self.dropout(inputs)))
        out = self.fc_2(self.dropout(out))
        return out


class MixFC(Module):
    def __init__(self, d_model):
        super(MixFC, self).__init__()
        self.fc_1 = nn.Linear(d_model * 2, 2048)
        self.fc_2 = nn.Linear(2048, d_model)
        self.dropout = nn.Dropout(p=0.3)
        self.double()

    def forward(self, inputs):
        out = f.relu(self.fc_1(self.dropout(inputs)))
        out = self.fc_2(self.dropout(out))
        return out


class Transformer(Module):
    def __init__(self, num_embeddings, embedding_dim, d_model, nhead, num_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        self.fc_d = FinalFC(self.d_model)
        self.fc_h = FinalFC(self.d_model)
        self.fc_mix = MixFC(self.d_model)

        self.double()

    def forward(self, inputs, train=True):
        if train:
            src = self.embedding(inputs[:, :15]).permute(1, 0, 2)
            tgt = self.embedding(inputs[:, 15:]).view(-1, 1024)
            tgt = self.fc_mix(tgt).view(inputs.size(0), -1, 512).permute(1, 0, 2)
            src = self.transformer_encoder(src)
            out = self.transformer_decoder(tgt, src)

            out = out.view(-1, out.size(2))
            day = self.fc_d(out).view(inputs.size(0), -1)
            hour = self.fc_h(out).view(inputs.size(0), -1)
            out = torch.cat((day, hour), dim=1)

        else:
            src = self.embedding(inputs[:, :15])
            tgt = self.embedding(inputs[:, 15:])
            src = self.transformer_encoder(src)
            out = self.transformer_decoder(tgt, src)

            out = out.view(-1, out.size(2))
            out = f.relu(self.fc1(self.dropout(out)))
            out = self.fc2(self.dropout(out))
            out = out.view(inputs.size(0), -1)
        return out
