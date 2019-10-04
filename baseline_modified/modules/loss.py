import torch
from torch.nn import Module


class MSELoss(Module):
    def __init__(self):
        super().__init__()
        # self.early = early
        # self.gap = gap
        # self.late = late
        
    def forward(self, X, y, train=True):
        # if train:
        #     X = y - X
        #     weight = torch.zeros_like(X)
        #     for r in range(X.size(0)):
        #         if X[r] >= self.gap:
        #             weight[r] = self.early
        #        elif X[r] < 0:
        #             weight[r] = self.late
        #     return torch.mean((X * weight) ** 2)**0.5
        # else:
        #     return torch.mean((y - X) ** 2)**0.5
       
        if X.sum() > y.sum():
            return torch.mean(torch.pow((X - y), 2))**0.5 * 10
        else:
            return torch.mean(torch.pow((X - y), 2))**0.5