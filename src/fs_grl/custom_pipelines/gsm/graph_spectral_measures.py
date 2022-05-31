from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

from fs_grl.gsm.graph_ismorphism_network import GINClassifier, GraphIsomorphismNetwork
from fs_grl.gsm.modules import GATLayer, Regularizer
from fs_grl.gsm.utils import create_gat_knn_params


class GraphSpectralMeasures(nn.Module):
    """
    Model from "Few-Shot Learning on Graphs via Super-Classes based on Graph Spectral Measures"
    https://arxiv.org/abs/2002.12815
    """

    def __init__(
        self,
        model_cfg,
        feature_dim,
        hidden_dim,
        output_dim,
        num_layers,
        num_mlp_layers,
        final_dropout,
        learn_eps,
        graph_pooling_type,
        neighbor_pooling_type,
        gat_params,
        knn_value,
    ):
        super().__init__()

        gat_layer_params, knn_params = create_gat_knn_params(model_cfg)
        self.knn_params = knn_params
        self.knn_value = knn_value

        num_gat_layers = gat_params["num_gat_layers"]
        if gat_layer_params[num_gat_layers - 1]["concat"] == 0:
            self.final_gat_out_dim = gat_layer_params[num_gat_layers - 1]["out_channels"]
        else:
            self.final_gat_out_dim = (
                gat_layer_params[num_gat_layers - 1]["out_channels"] * gat_layer_params[num_gat_layers - 1]["heads"]
            )

        self.num_gat_layers = num_gat_layers
        self.gat_layer_params = gat_layer_params

        self.gin = GraphIsomorphismNetwork(
            num_layers,
            num_mlp_layers,
            feature_dim,
            hidden_dim,
            output_dim,
            final_dropout,
            learn_eps,
            graph_pooling_type,
            neighbor_pooling_type,
        )

        self.gin_classifier = GINClassifier(
            num_layers=num_layers,
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=gat_params.gat_out_dim,
            final_dropout=final_dropout,
        )

        self.gat_modules = torch.nn.ModuleList()
        for i in range(num_gat_layers):
            self.gat_modules.append(GATLayer(gat_layer_params[i], knn_params[i]))

        self.regularizer = Regularizer()

    def forward(self, batch, is_metatest=False):
        pooled_h_layers, node_embeds, Adj_block_idx = self.gin(batch)

        x = pooled_h_layers[-1]
        gin_preds = self.gin_classifier(pooled_h_layers)

        if not is_metatest:
            edges = self.create_train_knn_graph(x, batch)
        else:
            edges = self.create_test_knn_graph(x, batch, gin_preds)

        x = torch.cat(pooled_h_layers[1:], dim=1)
        gat_outs = []
        x = F.normalize(x, p=2, dim=1)

        for i in range(self.num_gat_layers):
            x = self.gat_modules[i](x, edges)
            x = F.normalize(x, p=2, dim=1)
            gat_outs.append(x)

        return x, (node_embeds, Adj_block_idx), gin_preds, edges

    def create_train_knn_graph(self, embeds: torch.Tensor, graph_data_list: List[Data]):
        """

        :param embeds:
        :param graph_data_list:

        :return
        """

        embedded_graphs_by_superclass = {}

        # for each superclass, contains a mapping local index --> index in the batch
        superclass_index_to_batch_index = {}

        for i, graph in enumerate(graph_data_list):

            super_class = graph.super_class.item()

            embedded_graphs_by_superclass.setdefault(super_class, []).append(embeds[i].unsqueeze(0))

            superclass_index_to_batch_index.setdefault(super_class, {})
            cumulative_num_graphs_supclass = len(superclass_index_to_batch_index[super_class])
            superclass_index_to_batch_index[super_class][cumulative_num_graphs_supclass] = i

        all_edges = []

        for supercls, supercls_embeds in embedded_graphs_by_superclass.items():

            superclass_knn_graph = self.get_super_class_graph(
                supercls, supercls_embeds, superclass_index_to_batch_index
            )

            all_edges.append(superclass_knn_graph)

        return torch.cat(all_edges, dim=1)

    def get_super_class_graph(self, supercls, supercls_embeds, superclass_index_to_batch_index):
        supercls_embeds = torch.cat(supercls_embeds, dim=0)

        # knn graph between elements of the same superclass
        super_class_knn = knn_graph(supercls_embeds, self.knn_value, loop=True)

        # map back to batch-level index
        actual_super_class_knn = torch.zeros((super_class_knn.shape[0], super_class_knn.shape[1])).long()
        for i in range(super_class_knn.shape[0]):
            for j in range(super_class_knn.shape[1]):
                ind = int(super_class_knn[i, j].cpu().numpy())
                actual_super_class_knn[i, j] = superclass_index_to_batch_index[supercls][ind]

        return torch.LongTensor(actual_super_class_knn).type_as(supercls_embeds).long()

    def create_test_knn_graph(self, embeds, graph_data_list: List[Data], gin_preds):
        """

        :param embeds:
        :param graph_data_list:
        :param gin_preds:
        :return
        """

        super_class_segregation = {}
        superclass_index_to_batch_index = {}

        super_class_preds = torch.argmax(gin_preds, dim=1).cpu().numpy()

        for i, graph in enumerate(graph_data_list):

            if super_class_preds[i] not in super_class_segregation.keys():
                super_class_segregation[super_class_preds[i]] = []
            if super_class_preds[i] not in superclass_index_to_batch_index.keys():
                superclass_index_to_batch_index[super_class_preds[i]] = {}

            super_class_segregation[super_class_preds[i]].append(embeds[i].unsqueeze(0))
            superclass_index_to_batch_index[super_class_preds[i]][
                len(superclass_index_to_batch_index[super_class_preds[i]])
            ] = i

        all_edges = []

        for supercls, supercls_embeds in super_class_segregation.items():

            superclass_knn_graph = self.get_super_class_graph(
                supercls, supercls_embeds, superclass_index_to_batch_index
            )

            all_edges.append(superclass_knn_graph)

        return torch.cat(all_edges, dim=1)
