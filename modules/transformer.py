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

    def forward(self, inputs, field, train=True):
        if train:
            src = self.embedding(inputs[:, :14]).permute(1, 0, 2)
            src = self.transformer_encoder(src)

            tgt = self.embedding(inputs[:, 14:]).view(-1, 1024)
            tgt = self.fc_mix(tgt).view(inputs.size(0), -1, 512).permute(1, 0, 2)
            tgt_start = torch.zeros((1, inputs.size(0), 512), dtype=torch.double).cuda()
            tgt = torch.cat((tgt_start, tgt), dim=0)
            out = self.transformer_decoder(tgt, src)

            out = out.view(-1, out.size(2))
            day = self.fc_d(out).view(inputs.size(0), -1)
            hour = self.fc_h(out).view(inputs.size(0), -1)
            out = torch.cat((day, hour), dim=1)
            return out

        else:
            src = self.embedding(inputs[:, :14]).permute(1, 0, 2)
            src = self.transformer_encoder(src)

            tgt_start = torch.zeros((1, inputs.size(0), 512), dtype=torch.double).cuda()
            tgt = tgt_start
            result = []
            for i in range(4):
                out = self.transformer_decoder(tgt, src)
                out = out[-1, :, :]
                out = out.view(-1, out.size(-1))
                day = self.fc_d(out).view(inputs.size(0), -1)
                hour = self.fc_h(out).view(inputs.size(0), -1)

                result.append(day)
                result.append(hour)

                if i < 3:
                    day = self.embedding(self.time_to_idx(day, field, 'd', i).long()).squeeze(1)
                    hour = self.embedding(self.time_to_idx(hour, field, 'h').long()).squeeze(1)
                    new = torch.cat((day, hour), dim=-1).view(-1, 1024)
                    new = self.fc_mix(new).view(inputs.size(0), -1, 512).permute(1, 0, 2)
                    tgt = torch.cat((tgt, new))

            return result

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
