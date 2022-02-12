import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv.gat_conv import GATConv

from fs_grl.modules.mlp import MLP


class GNNEmbedder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, embedding_dim, num_mlp_layers):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.conv1 = GATConv(in_channels=self.feature_dim, out_channels=self.hidden_dim)
        self.conv2 = GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)

        self.mlp = MLP(
            num_layers=num_mlp_layers,
            input_dim=self.hidden_dim,
            output_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
        )

    def forward(self, batch):
        """
        Embeds a batch of graphs given as a single large graph

        :param batch: Batch containing graphs to embed
        :return: embedded graphs, each graph embedded as a point in R^{E}
        """

        # X ~ (num_nodes_in_batch, feature_dim)
        # edge_index ~ (2, num_edges_in_batch)
        X, edge_index = batch.x, batch.edge_index

        # h1 ~ (num_nodes_in_batch, hidden_dim)
        h1 = self.conv1(X, edge_index)
        h1 = F.relu(h1)

        # h2 ~ (num_nodes_in_batch, hidden_dim)
        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)

        # out ~ (num_nodes_in_batch, output_dim)
        node_out_features = self.mlp(h2)

        # pooled_out ~ (num_samples_in_batch, embedding_dim)
        pooled_out = global_mean_pool(node_out_features, batch.batch)
        return pooled_out
