import torch


def squared_l2(x, y):
    distances = torch.pow(x - y, 2).sum(-1)

    return distances


def squared_l2_similarity(x, y):
    distances = squared_l2(x, y)
    similarities = 1 / (1 + distances)
    return similarities
