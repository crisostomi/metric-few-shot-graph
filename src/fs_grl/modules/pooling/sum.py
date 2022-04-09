from torch import nn
from torch_geometric.nn import global_add_pool


class GlobalSumPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_add_pool(x=x, batch=batch)
