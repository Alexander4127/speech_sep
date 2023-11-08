import torch
import torch.nn as nn

from ss.utils.util import EPS


class GLayerNorm(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_channels, 1))
        self.beta = nn.Parameter(torch.zeros(n_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input with global layer normalization
        :param x: tensor with shape [N, C, T]
        :return:
        """
        mean = torch.mean(x, dim=(1, 2), keepdim=True)
        var = torch.var(x, dim=(1, 2), keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + EPS) + self.beta
