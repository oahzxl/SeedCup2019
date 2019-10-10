from torch.nn import Module
from torch import nn
import torch
import torch.nn.functional as f


class Simple(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Simple, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.encoder = nn.LSTM(input_size=300, hidden_size=300, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTMCell(input_size=600, hidden_size=600)

        self.fc_t_day = nn.Linear(in_features=600, out_features=1024)
        self.fc_t_day2 = nn.Linear(in_features=1024, out_features=1)
        self.fc_t_hour = nn.Linear(in_features=600, out_features=1024)
        self.fc_t_hour2 = nn.Linear(in_features=1024, out_features=1)

        self.dropout = nn.Dropout(p=0.5)
        self.double()

    def forward(self, inputs, mode, field):
        inputs = self.embedding(inputs)
        inputs, (_, _) = self.encoder(inputs)
        inputs = inputs[:, -1, :]

        outputs = []
        hx = torch.zeros_like(inputs)
        cx = torch.zeros_like(inputs)
        for i in range(4):
            hx, cx = self.decoder(inputs, (hx, cx))
            day = self.fc_t_day(f.relu(self.dropout(hx)))
            day = self.fc_t_day2(f.relu(self.dropout(day)))
            hour = self.fc_t_hour(f.relu(self.dropout(hx)))
            hour = self.fc_t_hour2(f.relu(self.dropout(hour)))
            outputs.append(day)
            outputs.append(hour)

            day = self.embedding(self.time_to_idx(day, field, 'd').long()).squeeze(1)
            hour = self.embedding(self.time_to_idx(hour, field, 't').long()).squeeze(1)
            inputs = torch.cat((day, hour), dim=1)

        return outputs

    @staticmethod
    def time_to_idx(time, field, mode):
        if mode == 'd':
            time = time * 4 + 2.5
            for b in range(time.size(0)):
                if time[b] > -4:
                    time[b] = field.vocab.stoi['%.0f' % time[b] + '_' + mode]
                else:
                    time[b] = field.vocab.stoi['-4' + '_' + mode]
        elif mode == 'h':
            time = time * 10 + 15
            for b in range(time.size(0)):
                time[b] = field.vocab.stoi['%.0f' % time[b] + '_' + mode]
        return time
