import torch
from torch.nn import Module


class RMSELoss(Module):
    def __init__(self, early, gap, late):
        super(RMSELoss, self).__init__()
        self.early = early
        self.gap = gap
        self.late = late

    def forward(self, inputs, targets, train=True):
        if train:
            inputs = targets - inputs
            weight = torch.ones_like(inputs)
            for r in range(inputs.size(0)):
                if inputs[r, 0] >= self.gap:
                    weight[r, 0] = self.early

                elif inputs[r, 0] < 0:
                    weight[r, 0] = self.late
            inputs = (inputs * weight) ** 2
        else:
            inputs = (targets - inputs) ** 2

        return torch.mean(inputs) ** 0.5
