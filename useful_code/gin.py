import torch
import torch.nn.functional as F
from torch import nn

from fs_grl.modules.mlp import MLP


class GINEmbedder(nn.Module):
    def __init__(
        self,
        num_layers,
        num_mlp_layers,
        feature_dim,
        hidden_dim,
        output_dim,
        final_dropout,
        learn_eps,
        graph_pooling_type,
        neighbor_pooling_type,
    ):
        """
        num_layers: number of layers in the neural networks (INCLUDING the input layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        final_dropout: dropout ratio on the final linear layer
        learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate
        neighbors and center nodes altogether.
        neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
        graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
        device: which device to use
        """

        super(GINEmbedder, self).__init__()

        self.final_dropout = final_dropout
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        self.MLPs = torch.nn.ModuleList()

        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            layer_input_dim = feature_dim if layer == 0 else hidden_dim
            self.MLPs.append(MLP(num_mlp_layers, layer_input_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function that maps the hidden representation at different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            layer_input_dim = feature_dim if layer == 0 else hidden_dim
            self.linears_prediction.append(nn.Linear(layer_input_dim, output_dim))

        self.embedding_dim = output_dim  # TODO: fix

    def __preprocess_neighbors_maxpool(self, batch_graph):
        # create padded_neighbor_list in concatenated graph

        # compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.highest_degree_node for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.x))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                # add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                # padding, dummy data is assumed to be stored in -1
                pad.extend([-1] * (max_deg - len(pad)))

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        # create block diagonal sparse matrix

        ref_tensor = batch_graph[0].x

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.x))
            edge_mat_list.append(graph.edge_index + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1).type_as(ref_tensor).long()
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1]).type_as(ref_tensor).long()

        Adj_block = torch.sparse.FloatTensor(
            Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]])
        ).float()

        return Adj_block

    def __preprocess_graphpool(self, batch_graph):
        # create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)

        start_idx = [0]
        ref_tensor = batch_graph[0].x
        # compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.x))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            if self.graph_pooling_type == "average":
                elem.extend([1.0 / len(graph.x)] * len(graph.x))
            else:
                # sum pooling
                elem.extend([1] * len(graph.x))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

        return graph_pool.cuda()

    def maxpool(self, h, padded_neighbor_list):
        # Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]
        return pooled_rep

    def next_layer_eps(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        # pooling neighboring nodes and center nodes separately by epsilon reweighting.

        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.MLPs[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def next_layer(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        # pooling neighboring nodes and center nodes altogether

        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # representation of neighboring and center nodes
        pooled_rep = self.MLPs[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def forward(self, batch_graph):
        X_concat = batch_graph.x
        batch_graph = batch_graph.to_data_list()

        graph_pool = self.__preprocess_graphpool(batch_graph)

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)

        # list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat

        for layer in range(self.num_layers - 1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block=Adj_block)

            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, h)
            score_over_layer += F.dropout(
                self.linears_prediction[layer](pooled_h), self.final_dropout, training=self.training
            )

        return score_over_layer
