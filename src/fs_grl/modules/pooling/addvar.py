import logging

import torch
from torch import nn
from torch_scatter import scatter_add, scatter_std

pylogger = logging.getLogger(__name__)


class GlobalAddVarPool(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.mid_features = out_features // 2
        self.out_features = out_features

        if self.out_features % 2 != 0:
            message = f"<AddVar> pooling requires an even number of output features, got '{self.out_features}'!"
            pylogger.error(message)
            raise ValueError(message)

        self.linear_projection = nn.Linear(in_features=self.in_features, out_features=self.mid_features)

    def forward(self, x, batch):
        features = self.linear_projection(x)
        var_features = scatter_std(src=features, index=batch, dim=0)
        add_features = scatter_add(src=features, index=batch, dim=0)
        out = torch.cat((add_features, var_features), dim=-1)
        return out
