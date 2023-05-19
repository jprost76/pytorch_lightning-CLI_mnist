import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, hidden_dim: int=10, out_dim: int=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        modules = [
            nn.Linear(28**2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)
