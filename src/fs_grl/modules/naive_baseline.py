from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv.gat_conv import GATConv

from fs_grl.data.episode import EpisodeBatch, EpisodeHParams
from fs_grl.modules.mlp import MLP


class GNNEncoder(nn.Module):
    def __init__(
        self, feature_dim, hidden_dim, output_dim, num_classes, num_mlp_layers, episode_hparams: EpisodeHParams
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.output_dim = output_dim

        self.conv1 = GATConv(in_channels=self.feature_dim, out_channels=self.hidden_dim)
        self.conv2 = GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.episode_hparams = episode_hparams

        self.mlp = MLP(
            num_layers=num_mlp_layers,
            input_dim=self.hidden_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
        )

        self.num_supports_per_episode = (
            self.episode_hparams.num_supports_per_class * self.episode_hparams.num_classes_per_episode
        )
        self.num_queries_per_episode = (
            self.episode_hparams.num_queries_per_class * self.episode_hparams.num_classes_per_episode
        )

    def embed_supports(self, supports: Batch):
        """
        :param supports: Batch containing BxNxK support graphs as a single large graph
        :return: embedded supports ~ ((B*N*K)xE), each graph embedded as a point in R^{E}
        """
        return self._embed(supports)

    def embed_queries(self, queries: Batch):
        """
        :param queries: Batch containing BxNxQ query graphs
        :return: embedded queries ~ (BxNxQxE), each graph embedded as a point in R^{E}
        """
        return self._embed(queries)

    def _embed(self, batch: Batch):
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

    def get_class_prototypes(self, embedded_supports: torch.Tensor, batch: EpisodeBatch):
        """

        :param embedded_supports: tensor ~ (num_supports_batch, embedding_dim)
        :return:
        """
        device = embedded_supports.device
        batch_size = batch.num_episodes

        # sequence of embedded supports for each episode, each has shape (num_supports_per_episode, hidden_dim)
        embedded_supports_per_episode = embedded_supports.split(tuple([self.num_supports_per_episode] * batch_size))

        # sequence of labels for each episode, each has shape (num_supports_per_episode)
        labels_per_episode = batch.supports.y.split(tuple([self.num_supports_per_episode] * batch_size))

        classes_per_episode = batch.labels.split([self.episode_hparams.num_classes_per_episode] * batch_size)

        all_class_prototypes = []
        for episode in range(batch_size):

            embedded_supports = embedded_supports_per_episode[episode]
            labels = labels_per_episode[episode]
            classes = classes_per_episode[episode]

            class_prototypes_episode = {}
            for cls in classes:
                class_indices = torch.arange(len(labels), device=device)[labels == cls]
                class_supports = torch.index_select(embedded_supports, dim=0, index=class_indices)
                class_prototypes = class_supports.mean(dim=0)
                class_prototypes_episode[cls.item()] = class_prototypes

            all_class_prototypes.append(class_prototypes_episode)

        return all_class_prototypes

    def align_queries_prototypes(self, batch, embedded_queries: torch.Tensor, class_prototypes: List[Dict]):
        """

        :param batch:
        :param embedded_queries: shape (num_queries_batch, hidden_dim)
        :param class_prototypes:
        :return:
        """
        device = embedded_queries.device
        batch_size = batch.num_episodes

        batch_queries = []
        batch_prototypes = []
        embedded_queries_per_episode = embedded_queries.split(tuple([self.num_queries_per_episode] * batch_size))

        for episode in range(batch_size):

            sorted_class_prototypes = [
                (global_class, prototype) for global_class, prototype in class_prototypes[episode].items()
            ]
            sorted_class_prototypes.sort(key=lambda tup: tup[0])
            sorted_class_prototypes_tensors = [tup[1] for tup in sorted_class_prototypes]

            # shape (num_classes_episode, hidden_dim)
            class_prototype_matrix = torch.stack(sorted_class_prototypes_tensors)

            # shape (num_queries_episode, hidden_dim)
            episode_embedded_queries = embedded_queries_per_episode[episode]

            repeated_embedded_queries = episode_embedded_queries.repeat_interleave(
                self.episode_hparams.num_classes_per_episode, dim=0
            )

            repeated_class_prototypes = class_prototype_matrix.repeat((self.num_queries_per_episode, 1))

            batch_queries.append(repeated_embedded_queries)
            batch_prototypes.append(repeated_class_prototypes)

        return torch.cat(batch_queries, dim=0), torch.cat(batch_prototypes, dim=0)

    def forward(self, batch: EpisodeBatch):
        """
        :param supports:
        :return:
        """

        supports, queries = batch.supports, batch.queries

        # shape (num_supports_batch, hidden_dim)
        embedded_supports = self.embed_supports(supports)

        # shape (num_queries_batch, hidden_dim)
        embedded_queries = self.embed_queries(queries)

        class_prototypes = self.get_class_prototypes(embedded_supports, batch)

        batch_queries, batch_prototypes = self.align_queries_prototypes(batch, embedded_queries, class_prototypes)

        return batch_queries, batch_prototypes
