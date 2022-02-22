import torch
from torch import nn

from fs_grl.modules.losses.utils import get_reduction


class LogisticLoss(nn.Module):
    def __init__(self, margin, reduction):
        super().__init__()
        self.margin = margin
        self.reduction = get_reduction(reduction)
        self.relu = nn.ReLU()

    def forward(self, similarities: torch.Tensor, targets: torch.Tensor):
        """
        When y = 1 the loss results in (1 - similarity), otherwise when y = -1 the loss is max(0, similarity - margin)
        """

        return torch.log(1 + torch.exp(-1 * targets * similarities))
