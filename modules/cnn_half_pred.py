import torch
import torch.nn.functional as f
from torch import nn
from torch.nn import Module

from modules.restnet import ResNet


class HalfPredCNN(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(HalfPredCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # self.cnn = CNNCell()   # with relu and bn
        self.cnn = ResNet(layers=[2, 2, 2, 2])

        self.fc_normal_1 = nn.Linear(in_features=6144, out_features=2048)
        self.fc_normal_2 = nn.Linear(in_features=2048, out_features=512)
        self.fc_pred_prov = nn.Linear(in_features=512, out_features=28)
        self.fc_pred_city = nn.Linear(in_features=512, out_features=120)
        self.fc_pred_lgst = nn.Linear(in_features=512, out_features=14)
        self.fc_pred_warehouse = nn.Linear(in_features=512, out_features=12)

        self.dropout = nn.Dropout(p=0.5)
        self.double()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, inputs, field, train=True):

        normal_data = self.embedding(inputs).unsqueeze(1)
        normal_data = self.cnn(normal_data.view(normal_data.size(0), normal_data.size(1), -1, 60))
        normal_data = self.dropout(normal_data.view(normal_data.size(0), -1))
        normal_data = f.relu(self.dropout(self.fc_normal_1(normal_data)))
        normal_data = f.relu(self.dropout(self.fc_normal_2(normal_data)))
        prov = self.fc_pred_prov(normal_data)
        city = self.fc_pred_city(normal_data)
        lgst = self.fc_pred_lgst(normal_data)
        warehouse = self.fc_pred_warehouse(normal_data)

        return prov, city, lgst, warehouse
