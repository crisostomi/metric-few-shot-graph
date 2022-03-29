import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, GraphNorm, JumpingKnowledge

from fs_grl.modules.mlp import MLP


class NodeEmbedder(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        embedding_dim,
        num_mlp_layers,
        num_convs,
        dropout_rate,
        do_preprocess,
        use_batch_norm=True,
        jump_mode="cat",
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_convs = num_convs
        self.jump_mode = jump_mode
        self.do_preprocess = do_preprocess
        self.use_batch_norm = use_batch_norm

        self.preprocess_mlp = (
            Sequential(
                Linear(self.feature_dim, self.hidden_dim),
                GraphNorm(self.hidden_dim) if self.use_batch_norm else nn.Identity(),
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
                    GraphNorm(self.hidden_dim) if self.use_batch_norm else nn.Identity(),
                    ReLU(),
                    Linear(self.hidden_dim, self.hidden_dim),
                    ReLU(),
                )
            )
            self.convs.append(conv)

        num_layers = (self.num_convs + 1) if self.do_preprocess else self.num_convs
        pooled_dim = (num_layers * self.hidden_dim) + self.feature_dim if self.jump_mode == "cat" else self.hidden_dim

        self.dropout = nn.Dropout(p=dropout_rate)

        self.mlp = MLP(
            num_layers=num_mlp_layers,
            input_dim=pooled_dim,
            output_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            use_batch_norm=self.use_batch_norm,
        )

        self.jumping_knowledge = JumpingKnowledge(mode=self.jump_mode) if self.jump_mode != "none" else None

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

        if self.jump_mode != "none":
            h = self.jumping_knowledge(jump_xs)

        h = self.dropout(h)
        # out ~ (num_nodes_in_batch, output_dim)
        node_out_features = self.mlp(h)

        return node_out_features
