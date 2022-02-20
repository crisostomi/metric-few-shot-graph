import abc
from typing import Dict, List

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch_geometric.data import Batch

from fs_grl.data.episode import EpisodeBatch


class GNNEmbeddingSimilarity(nn.Module, abc.ABC):
    def __init__(self, cfg, feature_dim, num_classes, **kwargs):
        super().__init__()
        self.cfg = cfg

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.embedder = instantiate(
            self.cfg.embedder,
            feature_dim=self.feature_dim,
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

        embedded_batch = self.embedder(batch)
        return embedded_batch

    def get_class_prototypes(self, embedded_supports: torch.Tensor, batch: EpisodeBatch):
        """
        Computes the prototype of each class as the mean of the embedded supports for that class

        :param batch:
        :param embedded_supports: tensor ~ (num_supports_batch, embedding_dim)
        :return:
        """
        device = embedded_supports.device
        num_episodes = batch.num_episodes

        num_supports_per_episode = (
            batch.episode_hparams.num_supports_per_class * batch.episode_hparams.num_classes_per_episode
        )

        # sequence of embedded supports for each episode, each has shape (num_supports_per_episode, hidden_dim)
        embedded_supports_per_episode = embedded_supports.split(tuple([num_supports_per_episode] * num_episodes))
        # sequence of labels for each episode, each has shape (num_supports_per_episode)
        labels_per_episode = batch.supports.y.split(tuple([num_supports_per_episode] * num_episodes))
        classes_per_episode = batch.global_labels.split([batch.episode_hparams.num_classes_per_episode] * num_episodes)

        all_class_prototypes = []
        for episode in range(num_episodes):

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

        num_queries_per_episode = (
            batch.episode_hparams.num_queries_per_class * batch.episode_hparams.num_classes_per_episode
        )

        batch_size = batch.num_episodes

        batch_queries = []
        batch_prototypes = []
        embedded_queries_per_episode = embedded_queries.split(tuple([num_queries_per_episode] * batch_size))

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
                batch.episode_hparams.num_classes_per_episode, dim=0
            )

            repeated_class_prototypes = class_prototype_matrix.repeat((num_queries_per_episode, 1))

            batch_queries.append(repeated_embedded_queries)
            batch_prototypes.append(repeated_class_prototypes)

        return torch.cat(batch_queries, dim=0), torch.cat(batch_prototypes, dim=0)

    @abc.abstractmethod
    def get_similarities(self, queries, prototypes):
        pass

    def forward(self, batch: EpisodeBatch):
        """
        :param batch:
        :return:
        """

        supports, queries = batch.supports, batch.queries

        # shape (num_supports_batch, hidden_dim)
        embedded_supports = self.embed_supports(supports)

        # shape (num_queries_batch, hidden_dim)
        embedded_queries = self.embed_queries(queries)

        class_prototypes = self.get_class_prototypes(embedded_supports, batch)

        # both shape (num_queries_batch*num_classes, hidden_dim)
        batch_queries, batch_prototypes = self.align_queries_prototypes(batch, embedded_queries, class_prototypes)

        similarities = self.get_similarities(batch_queries, batch_prototypes)

        return similarities
