import torch
import torch.nn.functional as F
from torch import nn


class PolyCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, epsilon=1.0):
        """

        :param logits:
        :param labels:
        :param epsilon:

        :return:
        """

        # pt, CE, and poly_loss have shape (B,)
        pt = torch.sum(F.one_hot(labels) * F.softmax(logits), dim=-1)

        cross_entropy_loss = F.cross_entropy(logits, labels, reduction="none")

        poly_loss = cross_entropy_loss + epsilon * (1 - pt)

        return poly_loss.mean()
