import torch
import torch.nn.functional as f
from torch import nn
from torch.nn import Module


class SimpleRNN(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.encoder = nn.LSTM(input_size=300, hidden_size=300, bidirectional=True, batch_first=True, dropout=0.1)
        self.decoder = nn.LSTMCell(input_size=600, hidden_size=600)

        self.fc_t_day = nn.Linear(in_features=600, out_features=1024)
        self.fc_t_day2 = nn.Linear(in_features=1024, out_features=1)
        self.fc_t_hour = nn.Linear(in_features=600, out_features=1024)
        self.fc_t_hour2 = nn.Linear(in_features=1024, out_features=1)

        self.dropout = nn.Dropout(p=0.5)
        self.double()

    def forward(self, inputs, mode, field):
        ip = inputs
        inputs = self.embedding(inputs[:, :15])
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

            if i != 3:
                if mode == 'train':
                    day = self.embedding(ip[:, 15 + 2 * i])
                    hour = self.embedding(ip[:, 15 + 2 * i + 1])
                else:
                    day = self.embedding(self.time_to_idx(day, field, 'd', i).long()).squeeze(1)
                    hour = self.embedding(self.time_to_idx(hour, field, 't').long()).squeeze(1)
                inputs = torch.cat((day, hour), dim=1)

        return outputs

    @staticmethod
    def time_to_idx(time, field, mode, idx=0):
        if mode == 'd':
            # plus_list = [0.5, 0.4, 0.4]
            # mul_list = [1, 1, 1]
            plus_list = [1, 1, 1]
            mul_list = [2, 2, 2]

            time = time * mul_list[idx] + plus_list[idx]
            for b in range(time.size(0)):
                if time[b] < 0:
                    time[b] = field.vocab.stoi['0' + '_' + mode]
                elif time[b] > 15:
                    time[b] = field.vocab.stoi['15' + '_' + mode]
                else:
                    time[b] = field.vocab.stoi['%.0f' % time[b] + '_' + mode]
        elif mode == 'h':
            time = time * 5 + 15
            for b in range(time.size(0)):
                time[b] = field.vocab.stoi['%.0f' % time[b] + '_' + mode]
        return time
