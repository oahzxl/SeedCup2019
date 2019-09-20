import torch
from torch.nn import Module
from torch import nn


class CNN(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.cnn_time = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=0),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=0),
            nn.MaxPool2d(2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=0),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=1, padding=0),
            nn.MaxPool2d(2, stride=1),
            nn.ReLU(),
        )
        self.cnn_pred = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 2), stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(2, 2), stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=1, padding=0),
            nn.MaxPool2d(2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=1, padding=0),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2), stride=1, padding=0),
            nn.MaxPool2d(2, stride=1),
            nn.ReLU(),
        )
        self.fc_out_pre = 512
        self.fc_pre = nn.Linear(in_features=5904, out_features=512)
        self.fc_lgst = nn.Linear(in_features=self.fc_out_pre, out_features=14)
        self.fc_warehouse = nn.Linear(in_features=self.fc_out_pre, out_features=12)
        self.fc_prov = nn.Linear(in_features=self.fc_out_pre, out_features=28)
        self.fc_city = nn.Linear(in_features=self.fc_out_pre, out_features=120)

        self.fc_in_t = 1024
        self.fc_t = nn.Linear(in_features=11136, out_features=1024)
        self.fc_t1 = nn.Linear(in_features=self.fc_in_t, out_features=1)
        self.fc_t2 = nn.Linear(in_features=self.fc_in_t, out_features=1)
        self.fc_t3 = nn.Linear(in_features=self.fc_in_t, out_features=1)
        self.fc_t4 = nn.Linear(in_features=self.fc_in_t, out_features=1)

        self.double()

    def forward(self, inputs, mode, field):
        inputs = self.embedding(inputs).unsqueeze(1)

        if mode == 'train':
            # predict lost information
            pred = inputs[:, :, :8, :]
            pred = self.cnn_pred(pred)
            pred = pred.view(inputs.size(0), -1)
            pred = self.fc_pre(pred)
            lgst = self.fc_lgst(pred)
            warehouse = self.fc_warehouse(pred)
            prov = self.fc_prov(pred)
            city = self.fc_city(pred)

            # predict final time
            inputs = self.cnn_time(inputs)
            inputs = inputs.view(inputs.size(0), -1)
            inputs = self.fc_t(inputs)
            t1 = self.fc_t1(inputs)
            t2 = self.fc_t2(inputs)
            t3 = self.fc_t3(inputs)
            t4 = self.fc_t4(inputs)

            return lgst, warehouse, prov, city, t1, t2, t3, t4

        else:
            pred = inputs[:, :, :8, :]
            pred = self.cnn_pred(pred)
            pred = pred.view(inputs.size(0), -1)
            pred = self.fc_pre(pred)
            lgst = torch.argmax(self.fc_lgst(pred), dim=1).unsqueeze(1)
            lgst = self.stoi(lgst, field, '12')
            warehouse = torch.argmax(self.fc_warehouse(pred), dim=1).unsqueeze(1)
            warehouse = self.stoi(warehouse, field, '13')
            prov = torch.argmax(self.fc_prov(pred), dim=1).unsqueeze(1)
            prov = self.stoi(prov, field, '14')
            city = torch.argmax(self.fc_city(pred), dim=1).unsqueeze(1)
            city = self.stoi(city, field, '15')
            pred = torch.cat((lgst, warehouse, prov, city), dim=1)
            pred = self.embedding(pred).unsqueeze(1)

            inputs = torch.cat((inputs, pred), dim=2)
            inputs = self.cnn_time(inputs)
            inputs = inputs.view(inputs.size(0), -1)
            inputs = self.fc_t(inputs)
            t4 = self.fc_t4(inputs)
            return t4

    @staticmethod
    def stoi(inputs, field, idx):
        new = torch.zeros_like(inputs)
        for i in range(inputs.size(0)):
            new[i, 0] = field.vocab.stoi[str(inputs[i, 0]) + '_' + idx]
        return new