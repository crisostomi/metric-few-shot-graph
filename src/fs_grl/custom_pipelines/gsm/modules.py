import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATLayer(nn.Module):
    def __init__(self, gat_params, knn_value):
        super().__init__()
        self.knn_value = knn_value
        in_channels = gat_params["in_channels"]
        out_channels = gat_params["out_channels"]
        num_heads = gat_params["heads"]
        concat = gat_params["concat"]  # bool value - whether to concat the multi-head embeddings
        leaky_slope = gat_params["leaky_slope"]
        dropout = gat_params["dropout"]

        self.gat_layer = GATConv(
            in_channels, out_channels, heads=num_heads, concat=concat, negative_slope=leaky_slope, dropout=dropout
        )

    def forward(self, x, edges):
        out_x = self.gat_layer(x, edges)
        return out_x


class ClassifierLayer(nn.Module):
    def __init__(self, type_, final_gat_out_dim, num_classes):
        super().__init__()
        self.type_ = type_

        if type_ == "linear":
            self.drop = nn.Dropout(0.5)
            self.linear_layer1 = nn.Linear(final_gat_out_dim, final_gat_out_dim // 2)
            self.linear_layer2 = nn.Linear(final_gat_out_dim // 2, num_classes)
        else:
            self.class_reps = nn.Parameter(torch.randn(final_gat_out_dim, num_classes))

    def forward(self, x):
        if self.type_ == "linear":
            out1 = self.linear_layer1(x)
            return self.linear_layer2(self.drop(out1)), out1
        else:
            self.class_reps.data = F.normalize(self.class_reps.data, p=2, dim=1)
            return torch.mm(x, self.class_reps)


class Regularizer:
    def __init__(self):
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_func = torch.nn.L1Loss()

    def forward(self, node_embeds, Adj_block_idx):

        n1 = node_embeds[Adj_block_idx[0]]
        n2 = node_embeds[Adj_block_idx[1]]

        ones = torch.ones(Adj_block_idx.shape[1]).cuda()

        dot_prod = self.sigmoid(torch.sum(n1 * n2, dim=1))
        rec_loss = self.loss_func(dot_prod, ones)

        return rec_loss
