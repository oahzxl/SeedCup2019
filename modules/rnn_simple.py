import torch
import torch.nn.functional as f
from torch import nn
from torch.nn import Module


class SimpleRNN(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.encoder = nn.LSTM(input_size=512, hidden_size=512, bidirectional=True, batch_first=True,
                               num_layers=2, dropout=0.2)
        self.decoder = nn.LSTMCell(input_size=512, hidden_size=512)

        self.fc_1 = nn.Linear(in_features=1024, out_features=2048)
        self.fc_2 = nn.Linear(in_features=2048, out_features=512)

        self.fc_t_day = nn.Linear(in_features=512, out_features=128)
        self.fc_t_day2 = nn.Linear(in_features=128, out_features=1)
        # self.fc_t_hour = nn.Linear(in_features=1024, out_features=128)
        # self.fc_t_hour2 = nn.Linear(in_features=128, out_features=1)

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
            outputs.append(day)

            if i != 3:
                day = self.embedding(self.time_to_idx(day, field, 'd', i).long()).squeeze(1)
                inputs = day

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
                time[b] = field.vocab.stoi['%.0f' % time[b] + '_' + mode]
        elif mode == 'h':
            time = time * 5 + 15
            for b in range(time.size(0)):
                time[b] = field.vocab.stoi['%.0f' % time[b] + '_' + mode]
        return time
