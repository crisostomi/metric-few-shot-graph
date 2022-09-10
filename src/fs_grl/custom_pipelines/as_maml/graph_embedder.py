import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter_add

from fs_grl.custom_pipelines.as_maml.layers import LinearFw
from fs_grl.custom_pipelines.as_maml.sag_pooling import SAGPooling
from fs_grl.custom_pipelines.as_maml.sage_conv import SAGEConv


class NodeInformationScore(MessagePassing):
    def __init__(self, improved=False, cached=False, **kwargs):
        super(NodeInformationScore, self).__init__(aggr="add", **kwargs)

        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        edge_index, _ = remove_self_loops(edge_index)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, 0, num_nodes)

        row, col = edge_index
        expand_deg = torch.zeros((edge_weight.size(0),), dtype=dtype, device=edge_index.device)
        expand_deg[-num_nodes:] = torch.ones((num_nodes,), dtype=dtype, device=edge_index.device)

        return (
            edge_index,
            expand_deg - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col],
        )

    def forward(self, x, edge_index, edge_weight=None):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    "Cached {} number of edges, but found {}".format(self.cached_num_edges, edge_index.size(1))
                )

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class GraphEmbedder(torch.nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_dim, pooling_ratio, dropout_ratio):
        super(GraphEmbedder, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio

        self.conv1 = SAGEConv(self.feature_dim, self.hidden_dim)
        self.conv2 = SAGEConv(self.hidden_dim, self.hidden_dim)
        self.conv3 = SAGEConv(self.hidden_dim, self.hidden_dim)

        self.calc_information_score = NodeInformationScore()

        self.pool1 = SAGPooling(self.hidden_dim, self.pooling_ratio)
        self.pool2 = SAGPooling(self.hidden_dim, self.pooling_ratio)
        self.pool3 = SAGPooling(self.hidden_dim, self.pooling_ratio)

        self.lin1 = LinearFw(self.hidden_dim * 2, self.hidden_dim)
        self.lin2 = LinearFw(self.hidden_dim, self.hidden_dim // 2)
        self.lin3 = LinearFw(self.hidden_dim // 2, self.num_classes)

        self.relu = F.leaky_relu

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        # edge_index = edge_index.transpose(0, 1)

        x = self.relu(self.conv1(x, edge_index, edge_attr), negative_slope=0.1)
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.relu(self.conv2(x, edge_index, edge_attr), negative_slope=0.1)
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, None, batch)

        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.relu(self.conv3(x, edge_index, edge_attr), negative_slope=0.1)
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x_information_score = self.calc_information_score(x, edge_index)
        score = torch.sum(torch.abs(x_information_score), dim=1)

        x = self.relu(x1, negative_slope=0.1) + self.relu(x2, negative_slope=0.1) + self.relu(x3, negative_slope=0.1)

        graph_emb = x

        # added to avoid features exploding over the Reddit dataset
        x = torch.nn.functional.normalize(x, p=2, dim=-1)

        x = self.relu(self.lin1(x), negative_slope=0.1)
        x = self.relu(self.lin2(x), negative_slope=0.1)
        x = self.lin3(x)

        return x, score.mean(), graph_emb
