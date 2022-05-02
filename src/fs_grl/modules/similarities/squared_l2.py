import torch


def squared_l2(x, y):
    distances = torch.pow(x - y, 2).sum(-1)

    return distances
