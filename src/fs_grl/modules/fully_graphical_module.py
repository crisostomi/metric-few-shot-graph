import abc
from typing import Dict, List

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch_geometric.data import Batch
from torch_geometric.nn import to_hetero

from fs_grl.data.episode import EpisodeBatch
from fs_grl.modules.losses.margin import MarginLoss
from fs_grl.modules.similarities.cosine import cosine


class FullyGraphicalModule(nn.Module, abc.ABC):
    def __init__(self, cfg, feature_dim, num_classes, margin, heterogeneous_metadata, **kwargs):
        super().__init__()
        self.cfg = cfg

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        embedder = instantiate(
            self.cfg.embedder,
            feature_dim=self.feature_dim,
        )
        # heterogeneous_metadata = tuple(heterogeneous_metadata)
        self.embedder = to_hetero(
            embedder,
            (["nodes"], [("nodes", "edges", "nodes"), ("nodes", "is_aggregated", "nodes")]),
            aggr="sum",
        )

        self.loss_func = MarginLoss(margin=margin, reduction="mean")

    def forward(self, batch: EpisodeBatch):

        # shape (num_support_nodes, embedding_dim)
        embedded_supports = self.embed_supports(batch.supports)

        embedded_queries = self.embed_queries(batch.queries)

        embedded_supports_aggregator = embedded_supports["nodes"][batch.supports["nodes"].ptr[1:] - 1]
        embedded_queries_aggregator = embedded_queries["nodes"][batch.queries["nodes"].ptr[1:] - 1]

        class_prototypes = self.get_class_prototypes(embedded_supports_aggregator, batch=batch)

        similarities = self.get_similarities(embedded_queries_aggregator, class_prototypes, batch)

        return {
            "embedded_queries": embedded_queries_aggregator,
            "embedded_supports": embedded_supports_aggregator,
            "class_prototypes": class_prototypes,
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

        embedded_batch = self.embedder(batch.x_dict, batch.edge_index_dict)
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

    def get_class_prototypes(self, aggregator_supports: torch.Tensor, batch):
        """
        Return a list (num_episode) of dictionaries where each dictionary maps the global labels of an episode to the
        corresponding embedded prototype nodes
        :param aggregators_supports:
        :param batch:
        """

        assert aggregator_supports.shape[0] == batch.supports["nodes"].y.shape[0]

        device = aggregator_supports.device

        aggregators_per_episode = aggregator_supports.split(
            tuple([batch.num_supports_per_episode] * batch.num_episodes)
        )
        labels_per_episode = batch.supports["nodes"].y.split(
            tuple([batch.num_supports_per_episode] * batch.num_episodes)
        )
        classes_per_episode = batch.global_labels.split(
            [batch.episode_hparams.num_classes_per_episode] * batch.num_episodes
        )

        all_class_prototypes = []
        for episode in range(batch.num_episodes):

            embedded_aggregator = aggregators_per_episode[episode]
            labels = labels_per_episode[episode]
            classes = classes_per_episode[episode]

            class_prototypes_episode = {}
            for cls in classes:
                class_indices = torch.arange(len(labels), device=device)[labels == cls]
                class_supports = torch.index_select(embedded_aggregator, dim=0, index=class_indices)
                class_prototypes = class_supports.mean(dim=0)
                class_prototypes_episode[cls.item()] = class_prototypes

            all_class_prototypes.append(class_prototypes_episode)

        return all_class_prototypes

    def align_queries_prototypes(
        self, batch, embedded_queries: torch.Tensor, class_prototypes: List[Dict[int, torch.Tensor]]
    ):
        """
        :param batch:
        :param embedded_queries: shape (num_queries_batch, hidden_dim)
        :param class_prototypes:
        :return:
        """

        num_queries_per_episode = (
            batch.episode_hparams.num_queries_per_class * batch.episode_hparams.num_classes_per_episode
        )

        num_episodes = batch.num_episodes

        batch_queries = []
        batch_prototypes = []
        embedded_queries_per_episode = embedded_queries.split(tuple([num_queries_per_episode] * num_episodes))

        for episode in range(num_episodes):

            class_prototype_matrix = self.get_prototype_matrix_from_dict(class_prototypes[episode])

            aligned_queries, aligned_prototypes = self.align_queries_prototypes_pairs(
                embedded_queries_per_episode[episode], class_prototype_matrix, batch
            )

            batch_queries.append(aligned_queries)
            batch_prototypes.append(aligned_prototypes)

        return {"queries": torch.cat(batch_queries, dim=0), "prototypes": torch.cat(batch_prototypes, dim=0)}

    def align_queries_prototypes_pairs(self, queries, prototypes_matrix, batch):
        # shape (num_queries_episode, hidden_dim)

        aligned_embedded_queries = queries.repeat_interleave(batch.episode_hparams.num_classes_per_episode, dim=0)

        aligned_prototypes = prototypes_matrix.repeat((batch.num_queries_per_episode, 1))

        return aligned_embedded_queries, aligned_prototypes

    @classmethod
    def get_prototype_matrix_from_dict(cls, class_prototypes):
        sorted_class_prototypes = [(global_class, prototype) for global_class, prototype in class_prototypes.items()]
        sorted_class_prototypes.sort(key=lambda tup: tup[0])
        sorted_class_prototypes_tensors = [tup[1] for tup in sorted_class_prototypes]

        # shape (num_classes_episode, hidden_dim)
        class_prototype_matrix = torch.stack(sorted_class_prototypes_tensors)

        return class_prototype_matrix

    def get_prediction_similarities(self, embedded_queries: torch.Tensor, class_prototypes, batch: EpisodeBatch):
        """

        :param embedded_queries:
        :param class_prototypes: list (num_episodes) of dictionaries, each dictionary maps the global labels of
                                 an episode to the corresponding prototype indices
        :param batch:
        :return:
        """

        # batch_queries_prototypes = self.align_queries_prototypes(batch, embedded_queries, class_prototypes)
        # batch_queries, batch_prototypes = batch_queries_prototypes["queries"], batch_queries_prototypes["prototypes"]

        # distances = torch.pow(torch.norm(batch_queries - batch_prototypes, p=2, dim=-1), 2)
        # similarities = 1 / (1 + distances)
        similarities = self.get_similarities(embedded_queries, class_prototypes, batch)
        return similarities

    def get_intra_class_variance(
        self, embedded_supports: torch.Tensor, label_to_prototype_embed_map: Dict, batch: EpisodeBatch
    ):
        """
        Computes the mean of the intra-class variance for each episode, which is the
        sum of the squared l2 distance between support aggregators and their corresponding class prototypes

        :param embedded_supports:
        :param label_to_prototype_embed_map:
        :param batch:
        :return:
        """
        support_aggregator_indices = batch.get_aggregator_indices("supports")

        # shape (B*N*K, embedding_dim)
        support_aggregators = embedded_supports[support_aggregator_indices]

        support_aggregators_by_episode = support_aggregators.split(
            [batch.num_supports_per_episode] * batch.num_episodes
        )
        labels_by_episode = batch.supports.y.split([batch.num_supports_per_episode] * batch.num_episodes)
        global_labels_by_episode = batch.global_labels.split(
            [batch.episode_hparams.num_classes_per_episode] * batch.num_episodes
        )

        inter_class_var = 0
        for episode_ind in range(batch.num_episodes):
            episode_global_labels = global_labels_by_episode[episode_ind]
            episode_labels = labels_by_episode[episode_ind]
            episode_label_to_prot_embed_map = label_to_prototype_embed_map[episode_ind]
            episode_support_aggregators = support_aggregators_by_episode[episode_ind]

            global_to_local_mapping = {
                global_label.item(): ind for ind, global_label in enumerate(episode_global_labels)
            }

            # map global labels to local labels for the support samples
            episode_local_labels = torch.tensor([global_to_local_mapping[label.item()] for label in episode_labels])

            # (N, embedding_dim)
            class_prototype_matrix = self.get_prototype_matrix_from_dict(episode_label_to_prot_embed_map)

            # (N*K, embedding_dim)
            aligned_class_prototypes = class_prototype_matrix[episode_local_labels]
            distance_from_prototype = torch.pow(
                torch.norm(episode_support_aggregators - aligned_class_prototypes, p=2, dim=-1), 2
            )
            inter_class_var += distance_from_prototype.sum(dim=-1)

        inter_class_var = inter_class_var / batch.num_episodes

        return inter_class_var
