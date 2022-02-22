import torch
from torch.nn import functional as F


def cosine(x, y):
    similarities = torch.einsum("qh,qh->q", (F.normalize(x), F.normalize(y)))
    return similarities
