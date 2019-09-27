import torch
import torch.nn.functional as f
from torch import nn
from torch.nn import Module

from modules.cnn import CNNCell


class TripleCNN(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(TripleCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.cnn = CNNCell()   # with relu and bn

        self.fc_normal_1 = nn.Linear(in_features=2240, out_features=512)
        self.fc_normal_2 = nn.Linear(in_features=512, out_features=512)
        self.fc_pred_prov = nn.Linear(in_features=512, out_features=28)
        self.fc_pred_city = nn.Linear(in_features=512, out_features=120)
        self.fc_pred_lgst = nn.Linear(in_features=512, out_features=14)
        self.fc_pred_warehouse = nn.Linear(in_features=512, out_features=12)

        self.fc_hidden_1 = nn.Linear(in_features=160, out_features=512)
        self.fc_hidden_2 = nn.Linear(in_features=512 * 2, out_features=512)
        self.fc_pred_shipped_day = nn.Linear(in_features=512, out_features=1)
        self.fc_pred_shipped_hour = nn.Linear(in_features=512, out_features=1)
        self.fc_pred_got_day = nn.Linear(in_features=512, out_features=1)
        self.fc_pred_got_hour = nn.Linear(in_features=512, out_features=1)
        self.fc_pred_dlved_day = nn.Linear(in_features=512, out_features=1)
        self.fc_pred_dlved_hour = nn.Linear(in_features=512, out_features=1)

        self.fc_time_1 = nn.Linear(in_features=320, out_features=512)
        self.fc_time_2 = nn.Linear(in_features=512 * 3, out_features=512)
        self.fc_pred_sign_day = nn.Linear(in_features=512, out_features=1)
        self.fc_pred_sign_hour = nn.Linear(in_features=512, out_features=1)

        self.dropout = nn.Dropout(p=0.5)
        self.double()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, inputs, field, train=True):

        normal_data = self.embedding(inputs[:, :15]).unsqueeze(1)
        normal_data = self.cnn(normal_data.view(normal_data.size(0), normal_data.size(1), -1, 60))
        normal_data = normal_data.view(normal_data.size(0), -1)
        normal_data = f.relu(self.dropout(self.fc_normal_1(normal_data)))
        normal_data = f.relu(self.dropout(self.fc_normal_2(normal_data)))
        prov = self.fc_pred_prov(normal_data)
        city = self.fc_pred_city(normal_data)
        lgst = self.fc_pred_lgst(normal_data)
        warehouse = self.fc_pred_warehouse(normal_data)

        if train:
            hidden_data = self.embedding(inputs[:, 15:19]).unsqueeze(1)
        else:
            lgst = self.stoi(lgst, field, '12')
            warehouse = self.stoi(warehouse, field, '13')
            prov = self.stoi(prov, field, '14')
            city = self.stoi(city, field, '15')
            hidden_data = torch.cat((lgst, warehouse, prov, city), dim=1).unsqueeze(1)
        hidden_data = self.cnn(hidden_data.view(hidden_data.size(0), hidden_data.size(1), -1, 30))
        hidden_data = hidden_data.view(hidden_data.size(0), -1)
        hidden_data = f.relu(self.dropout(self.fc_hidden_1(hidden_data)))
        hidden_data = torch.cat((normal_data, hidden_data), dim=1)
        hidden_data = f.relu(self.dropout(self.fc_hidden_2(hidden_data)))
        shipped_day = self.fc_pred_shipped_day(hidden_data)
        shipped_hour = self.fc_pred_shipped_hour(hidden_data)
        got_day = self.fc_pred_got_day(hidden_data)
        got_hour = self.fc_pred_got_hour(hidden_data)
        dlved_day = self.fc_pred_dlved_day(hidden_data)
        dlved_hour = self.fc_pred_dlved_hour(hidden_data)

        if train:
            time_data = self.embedding(inputs[:, 19:]).unsqueeze(1)
        else:
            shipped_day = self.stoi(shipped_day, field, 'd')
            shipped_hour = self.stoi(shipped_hour, field, 'h')
            got_day = self.stoi(got_day, field, 'd')
            got_hour = self.stoi(got_hour, field, 'h')
            dlved_day = self.stoi(dlved_day, field, 'd')
            dlved_hour = self.stoi(dlved_hour, field, 'h')
            time_data = torch.cat((shipped_day, shipped_hour, got_day, got_hour,
                                   dlved_day, dlved_hour), dim=1).unsqueeze(1)
        time_data = self.cnn(time_data.view(time_data.size(0), time_data.size(1), -1, 60))
        time_data = time_data.view(time_data.size(0), -1)
        time_data = f.relu(self.dropout(self.fc_time_1(time_data)))
        time_data = torch.cat((normal_data, hidden_data, time_data), dim=1)
        time_data = f.relu(self.dropout(self.fc_time_2(time_data)))
        sign_day = self.fc_pred_sign_day(time_data)
        sign_hour = self.fc_pred_sign_hour(time_data)

        return (prov, city, lgst, warehouse,
                shipped_day, shipped_hour, got_day, got_hour, dlved_day, dlved_hour,
                sign_day, sign_hour)

    def stoi(self, inputs, field, idx):
        """
        map string to embed idx
        """
        if idx == 'd':
            for b in range(inputs.size(0)):
                inputs[b] = field.vocab.stoi['%.0f' % (inputs[b] * 8 + 3) + '_' + idx]
        elif idx == 'h':
            for b in range(inputs.size(0)):
                inputs[b] = field.vocab.stoi['%.0f' % (inputs[b] * 10 + 15) + '_' + idx]
        else:
            inputs = torch.argmax(inputs, dim=1)
            for b in range(inputs.size(0)):
                inputs[b] = field.vocab.stoi[str(int(inputs[b])) + '_' + idx]
                
        inputs = self.embedding(inputs.long())
        return inputs
