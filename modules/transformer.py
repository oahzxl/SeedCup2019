import torch
import torch.nn.functional as f
from torch import nn
from torch.nn import Module


class FinalFC(Module):
    def __init__(self, d_model):
        super(FinalFC, self).__init__()
        self.fc_1 = nn.Linear(d_model, 256)
        self.fc_2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.double()

    def forward(self, inputs):
        out = f.relu(self.fc_1(self.dropout(inputs)))
        out = self.fc_2(self.dropout(out))
        return out


class MixFC(Module):
    def __init__(self, d_model):
        super(MixFC, self).__init__()
        self.fc_1 = nn.Linear(d_model * 2, 1024)
        self.fc_2 = nn.Linear(1024, d_model)
        self.dropout = nn.Dropout(p=0.5)
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

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.fc = FinalFC(14 * self.d_model)

        self.double()

    def forward(self, inputs, field, train=True):
        src = self.embedding(inputs).permute(1, 0, 2).cuda()
        src = self.transformer_encoder(src)

        day = self.fc(src.view(inputs.size(0), -1))
        return day

    @staticmethod
    def time_to_idx(time, field, mode, idx=0):
        if mode == 'd':
            # plus_list = [0.5, 0.4, 0.4]
            # mul_list = [1, 1, 1]
            plus_list = [1, 1, 1]
            mul_list = [2, 2, 2]

            time = time * mul_list[idx] + plus_list[idx]
            for b in range(time.size(0)):
                time[b] = field.vocab.stoi['%.0f' % time[b] + '_' + mode]
        elif mode == 'h':
            time = time * 5 + 15
            for b in range(time.size(0)):
                time[b] = field.vocab.stoi['%.0f' % time[b] + '_' + mode]
        return time

    @staticmethod
    def make_mask():

        tgt_mask = torch.zeros((4, 4), dtype=torch.double)

        for i in range(1, 4):
            for j in range(i):
                tgt_mask[i, j] = float('-inf')

        return tgt_mask
