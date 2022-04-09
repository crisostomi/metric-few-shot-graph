from torch import nn
from torch_geometric.nn import global_mean_pool


class GlobalMeanPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_mean_pool(x=x, batch=batch)
