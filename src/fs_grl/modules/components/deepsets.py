import torch
import torch.nn as nn


class DeepSetsEmbedder(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute the representation for each data point
        x = self.phi(x)

        # sum up the representations
        x = torch.sum(x, dim=0)

        # compute the output
        out = self.rho(x)

        return out
