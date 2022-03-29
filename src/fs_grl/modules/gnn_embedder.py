import logging

import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, GraphMultisetTransformer, JumpingKnowledge, global_add_pool
from torch_scatter import scatter_add, scatter_std

from fs_grl.modules.mlp import MLP

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


class GNNEmbedder(nn.Module):
    def __init__(
        self, feature_dim, hidden_dim, embedding_dim, num_mlp_layers, num_convs, dropout_rate, pooling, do_preprocess
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_convs = num_convs
        self.pooling_method = pooling
        self.do_preprocess = do_preprocess

        self.preprocess_mlp = (
            Sequential(
                Linear(self.feature_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Linear(self.hidden_dim, self.hidden_dim),
                ReLU(),
            )
            if do_preprocess
            else None
        )

        self.convs = nn.ModuleList()
        for conv in range(self.num_convs):
            input_dim = self.feature_dim if (conv == 0 and not self.do_preprocess) else self.hidden_dim
            conv = GINConv(
                Sequential(
                    Linear(input_dim, self.hidden_dim),
                    BatchNorm1d(self.hidden_dim),
                    ReLU(),
                    Linear(self.hidden_dim, self.hidden_dim),
                    ReLU(),
                )
            )
            self.convs.append(conv)

        self.jump_mode = "cat"

        num_layers = (self.num_convs + 1) if self.do_preprocess else self.num_convs
        pooled_dim = (num_layers * self.hidden_dim) + self.feature_dim if self.jump_mode == "cat" else self.hidden_dim

        self.dropout = nn.Dropout(p=dropout_rate)

        self.mlp = MLP(
            num_layers=num_mlp_layers,
            input_dim=pooled_dim,
            output_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
        )

        if self.pooling_method == "GMT":
            self.pooling = GraphMultisetTransformer(
                in_channels=self.embedding_dim, hidden_channels=self.embedding_dim, out_channels=self.embedding_dim
            )
        elif self.pooling_method == "ADDVAR":
            self.pooling = GlobalAddVarPool(in_features=self.embedding_dim, out_features=self.embedding_dim)
        else:
            self.pooling = global_add_pool

        self.jumping_knowledge = JumpingKnowledge(mode="cat")

    def forward(self, batch):
        """
        Embeds a batch of graphs given as a single large graph

        :param batch: Batch containing graphs to embed
        :return: embedded graphs, each graph embedded as a point in R^{E}
        """

        # X ~ (num_nodes_in_batch, feature_dim)
        # edge_index ~ (2, num_edges_in_batch)
        X, edge_index = batch.x, batch.edge_index

        h = self.preprocess_mlp(X) if self.do_preprocess else X
        jump_xs = [X, h] if self.do_preprocess else [X]

        for conv in self.convs:
            h = conv(h, edge_index)
            jump_xs.append(h)

        h = self.jumping_knowledge(jump_xs)

        # out ~ (num_nodes_in_batch, output_dim)
        node_out_features = self.mlp(h)
        node_out_features = self.dropout(node_out_features)

        # pooled_out ~ (num_samples_in_batch, embedding_dim)
        pooling_args = self.get_pooling_args(node_out_features, batch)
        pooled_out = self.pooling(**pooling_args)

        return pooled_out

    def get_pooling_args(self, node_out_features, batch):
        pooling_args = {"x": node_out_features, "batch": batch.batch}
        if self.pooling_method == "GMT":
            pooling_args["edge_index"] = batch.edge_index
        return pooling_args
