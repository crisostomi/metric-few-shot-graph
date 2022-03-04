import torch.nn as nn
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, GraphMultisetTransformer, JumpingKnowledge, global_add_pool

from fs_grl.modules.mlp import MLP


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

        self.pooling = (
            GraphMultisetTransformer(
                in_channels=self.embedding_dim, hidden_channels=self.embedding_dim, out_channels=self.embedding_dim
            )
            if self.pooling_method == "GMT"
            else global_add_pool
        )

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
            self.dropout(h)
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
