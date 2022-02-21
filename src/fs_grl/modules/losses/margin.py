import torch
from torch import nn


class MarginLoss(nn.Module):
    def __init__(self, margin, reduction):
        super().__init__()
        self.margin = margin
        self.reduction = self.get_reduction(reduction)
        self.relu = nn.ReLU()

    def forward(self, similarities, targets):
        """
        When y = 1 the loss results in (1 - similarity), otherwise when y = -1 the loss is max(0, similarity - margin)
        """
        x, y = similarities, targets
        return self.reduction((1 / 2) * (y + 1) * (y - x) + (1 / 2) * (1 - y) * self.relu(x - self.margin))

    def get_reduction(self, reduction: str):
        if reduction == "mean":
            return torch.mean
        if reduction == "sum":
            return torch.sum
        else:
            raise NotImplementedError(f"No such reduction: {reduction}")
