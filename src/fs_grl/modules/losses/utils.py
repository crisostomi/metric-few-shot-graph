import torch


def get_reduction(reduction: str):
    if reduction == "mean":
        return torch.mean
    if reduction == "sum":
        return torch.sum
    else:
        raise NotImplementedError(f"No such reduction: {reduction}")
