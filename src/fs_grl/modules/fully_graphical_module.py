import abc
from typing import Dict, List

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch_geometric.data import Batch

from fs_grl.data.episode import EpisodeBatch
from fs_grl.modules.losses.margin import MarginLoss
from fs_grl.modules.similarities.cosine import cosine


class FullyGraphicalModule(nn.Module, abc.ABC):
    def __init__(self, cfg, feature_dim, num_classes, margin, **kwargs):
        super().__init__()
        self.cfg = cfg

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.embedder = instantiate(
            self.cfg.embedder,
            feature_dim=self.feature_dim,
        )

        self.loss_func = MarginLoss(margin=margin, reduction="mean")

    def forward(self, batch: EpisodeBatch):

        # shape (num_support_nodes, embedding_dim)
        embedded_supports = self.embed_supports(batch.supports)

        label_to_prototype_embed_map = self.get_label_to_prototype_embedding_mapping(
            embedded_supports, batch.label_to_prototype_mapping
        )

        embedded_queries = self.embed_queries(batch.queries)

        query_aggregators = self.get_query_aggregator_embeddings(embedded_queries, batch)

        similarities = self.get_similarities(query_aggregators, label_to_prototype_embed_map, batch)

        return {
            "embedded_queries": embedded_queries,
            "class_prototypes": label_to_prototype_embed_map,
            "similarities": similarities,
        }

    def embed_supports(self, supports: Batch) -> torch.Tensor:
        """
        :param supports: Batch containing BxNxK support graphs as a single large graph
        :return tensor ~ (total_num_nodes, embedding_dim) embedding of each node in the support samples
        """

        return self._embed(supports)

    def embed_queries(self, queries: Batch) -> torch.Tensor:
        """
        :param queries: Batch containing BxNxQ query graphs
        :return tensor ~ (total_num_nodes, embedding_dim) embedding of each node in the query samples
        """
        return self._embed(queries)

    def _embed(self, batch: Batch):
        """
        Embeds a batch of graphs given as a single large graph

        :param batch: Batch containing graphs to embed
        :return: embedded nodes composing the batch of graphs
        """

        embedded_batch = self.embedder(batch)
        return embedded_batch

    def get_similarities(self, embedded_queries: torch.Tensor, class_prototypes, batch: EpisodeBatch):
        """

        :param embedded_queries:
        :param class_prototypes: list (num_episodes) of dictionaries, each dictionary maps the global labels of
                                 an episode to the corresponding prototype indices
        :param batch:
        :return:
        """

        batch_queries_prototypes = self.align_queries_prototypes(batch, embedded_queries, class_prototypes)
        batch_queries, batch_prototypes = batch_queries_prototypes["queries"], batch_queries_prototypes["prototypes"]

        similarities = cosine(batch_queries, batch_prototypes)

        return similarities

    def compute_loss(self, model_out, batch: EpisodeBatch, **kwargs):
        similarities = model_out["similarities"]

        return self.loss_func(similarities, batch.cosine_targets)

    def get_label_to_prototype_embedding_mapping(
        self, embedded_supports: torch.Tensor, label_to_prototype_idx_mapping: List[Dict]
    ):
        """
        Return a list (num_episode) of dictionaries where each dictionary maps the global labels of an episode to the
        corresponding embedded prototype nodes

        :param embedded_supports
        :param label_to_prototype_idx_mapping: list (num_episodes) of dictionaries, each dictionary maps the global labels of
                                               an episode to the corresponding prototype indices
        """
        return [
            {key: embedded_supports[value] for key, value in label_to_prototype_mapping_by_episode.items()}
            for label_to_prototype_mapping_by_episode in label_to_prototype_idx_mapping
        ]

    def get_query_aggregator_embeddings(self, embedded_queries: torch.Tensor, batch: EpisodeBatch) -> torch.Tensor:
        """
        Return the embedded query aggregators.
        :param embedded_queries: tensor ~ (B*N*Q, embedding_dim)
        :param batch:
        :return: embedded_aggregators: tensor ~ (B*N*Q*embedding_dim)
        """

        aggregator_indices = batch.get_aggregator_indices("queries")
        embedded_aggregators = embedded_queries[aggregator_indices]

        return embedded_aggregators

    def align_queries_prototypes(
        self, batch: EpisodeBatch, embedded_queries: torch.Tensor, class_prototypes: List[Dict[int, torch.Tensor]]
    ):
        """

        :param batch: EpisodeBatch
        :param embedded_queries: shape (num_queries_batch, hidden_dim)
        :param class_prototypes: list (num_episodes) containing for each episode
                                the mapping global label -> corresponding embedded prototype
        :return:
        """

        num_episodes = batch.num_episodes

        batch_queries = []
        batch_prototypes = []
        embedded_queries_per_episode = embedded_queries.split(tuple([batch.num_queries_per_episode] * num_episodes))

        for episode in range(num_episodes):

            class_prototype_matrix = self.get_prototype_matrix_from_dict(class_prototypes[episode])

            aligned_queries, aligned_prototypes = self.align_queries_prototypes_pairs(
                embedded_queries_per_episode[episode], class_prototype_matrix, batch
            )

            batch_queries.append(aligned_queries)
            batch_prototypes.append(aligned_prototypes)

        return {"queries": torch.cat(batch_queries, dim=0), "prototypes": torch.cat(batch_prototypes, dim=0)}

    def align_queries_prototypes_pairs(self, queries, prototypes_matrix, batch):
        """

        :param queries:
        :param prototypes_matrix:
        :param batch:
        :return:
        """
        # shape (num_queries_episode, hidden_dim)

        aligned_embedded_queries = queries.repeat_interleave(batch.episode_hparams.num_classes_per_episode, dim=0)

        aligned_prototypes = prototypes_matrix.repeat((batch.num_queries_per_episode, 1))

        return aligned_embedded_queries, aligned_prototypes

    @classmethod
    def get_prototype_matrix_from_dict(cls, label_to_prototype_embeddings: Dict) -> torch.Tensor:
        """
        Returns a matrix where row i contains the embedded class prototype
        for the i-th label in the sorted array of global labels

        :param label_to_prototype_embeddings: mapping global label -> corresponding embedding
        :return class_prototype_matrix: tensor (num_classes_episode, embedding_dim)
        """

        label_and_embedded_prototypes_tuples = [
            (global_class, prototype) for global_class, prototype in label_to_prototype_embeddings.items()
        ]
        label_and_embedded_prototypes_tuples.sort(key=lambda tup: tup[0])
        sorted_class_prototypes_tensors = [tup[1] for tup in label_and_embedded_prototypes_tuples]

        # shape (num_classes_episode, embedding_dim)
        class_prototype_matrix = torch.stack(sorted_class_prototypes_tensors)

        return class_prototype_matrix
