import torch
import torch.nn.functional as f
from torch import nn
from torch.nn import Module


class SimpleRNN(Module):
    def __init__(self, num_embedding, embedding_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)

        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim,
                               bidirectional=True, batch_first=True,
                               num_layers=2, dropout=0.1)
        self.decoder = nn.LSTMCell(input_size=embedding_dim * 2, hidden_size=embedding_dim * 2)

        self.fc_1 = nn.Linear(in_features=embedding_dim * 2, out_features=embedding_dim * 4)
        self.fc_2 = nn.Linear(in_features=embedding_dim * 4, out_features=embedding_dim * 2)

        self.fc_t_day = nn.Linear(in_features=embedding_dim * 2, out_features=embedding_dim)
        self.fc_t_day2 = nn.Linear(in_features=embedding_dim, out_features=1)
        self.fc_t_hour = nn.Linear(in_features=embedding_dim * 2, out_features=embedding_dim)
        self.fc_t_hour2 = nn.Linear(in_features=embedding_dim, out_features=1)

        self.dropout = nn.Dropout(p=0.5)
        self.double()

    def forward(self, inputs, mode, field):
        inputs = self.embedding(inputs)
        inputs, (_, _) = self.encoder(inputs)
        inputs = inputs[:, -1, :]
        inputs = self.fc_1(f.relu(self.dropout(inputs)))
        inputs = self.fc_2(f.relu(self.dropout(inputs)))

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
                day = self.embedding(self.time_to_idx(day, field, 'd', i).long()).squeeze(1)
                hour = self.embedding(self.time_to_idx(hour, field, 'h').long()).squeeze(1)
                inputs = torch.cat((day, hour), dim=1)

        return outputs

    @staticmethod
    def time_to_idx(time, field, mode, idx=0):
        if mode == 'd':
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
