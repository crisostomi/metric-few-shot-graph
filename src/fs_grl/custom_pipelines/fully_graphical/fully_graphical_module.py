from typing import Dict, List

import torch
from hydra.utils import instantiate

from fs_grl.data.episode.episode_batch import EpisodeBatch
from fs_grl.modules.baselines.prototypical_dml import PrototypicalDML
from fs_grl.modules.losses.margin import MarginLoss
from fs_grl.modules.similarities.cosine import cosine


class FullyGraphicalModule(PrototypicalDML):
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

        embedded_support_nodes = self.embed_supports(batch)
        # shape (num_support_nodes, embedding_dim)

        label_to_prototype_embed_map = self.get_prototypes(embedded_support_nodes, batch.label_to_prototype_mapping)

        embedded_queries = self.embed_queries(batch)

        similarities = self.get_similarities(embedded_queries, label_to_prototype_embed_map, batch)

        return {
            "embedded_queries": embedded_queries,
            "embedded_support_nodes": embedded_support_nodes,
            "class_prototypes": label_to_prototype_embed_map,
            "similarities": similarities,
        }

    def embed_queries(self, batch: EpisodeBatch):
        """
        :param batch: Batch containing BxNxQ query graphs
        :return embedded queries ~ (BxNxQxE), each graph embedded as a point in R^{E}
        """
        embedded_query_nodes = self._embed(batch.queries)
        embedded_queries = self.get_query_aggregator_embeddings(embedded_query_nodes, batch)

        return embedded_queries

    def embed_support_nodes(self, batch: EpisodeBatch):
        return self._embed(batch.supports)

    def embed_supports(self, batch: EpisodeBatch, embedded_support_nodes=None):
        embedded_support_nodes = (
            self.embed_support_nodes(batch) if embedded_support_nodes is None else embedded_support_nodes
        )
        support_aggregator_indices = batch.get_aggregator_indices("supports")

        # shape (B*N*K, embedding_dim)
        embedded_supports = embedded_support_nodes[support_aggregator_indices]
        return embedded_supports

    def get_similarities(self, embedded_queries: torch.Tensor, class_prototypes, batch: EpisodeBatch):
        """

        :param embedded_queries:
        :param class_prototypes: list (num_episodes) of dictionaries, each dictionary maps the global labels of
                                 an episode to the corresponding prototype indices
        :param batch:
        :return
        """

        batch_queries_prototypes = self.align_queries_prototypes(batch, embedded_queries, class_prototypes)
        batch_queries, batch_prototypes = batch_queries_prototypes["queries"], batch_queries_prototypes["prototypes"]

        similarities = cosine(batch_queries, batch_prototypes)

        return similarities

    def compute_loss(self, model_out, batch: EpisodeBatch, **kwargs):
        similarities = model_out["similarities"]

        return self.loss_func(similarities, batch.cosine_targets)

    def get_prototypes(self, embedded_supports: torch.Tensor, label_to_prototype_idx_mapping: List[Dict]):
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

    def get_query_aggregator_embeddings(self, embedded_query_nodes: torch.Tensor, batch: EpisodeBatch) -> torch.Tensor:
        """
        Return the embedded query aggregators.
        :param embedded_query_nodes:
        :param batch:
        :return embedded_aggregators: tensor ~ (B*N*Q*embedding_dim)
        """

        aggregator_indices = batch.get_aggregator_indices("queries")
        embedded_aggregators = embedded_query_nodes[aggregator_indices]

        return embedded_aggregators
