import torch
from torch import nn


class CamelModel(nn.Module):
    def __init__(self, num_classes):
        super(CamelModel, self).__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
