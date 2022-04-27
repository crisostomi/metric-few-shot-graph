import torch
from torch.nn import functional as F


def cosine(x, y):
    similarities = torch.einsum("qh,qh->q", (F.normalize(x, dim=-1), F.normalize(y, dim=-1)))
    return similarities


def cosine_distance_1D(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    similarity = F.cosine_similarity(x, y)
    similarity_normalized = (similarity + 1) / 2
    return 1 - similarity_normalized
