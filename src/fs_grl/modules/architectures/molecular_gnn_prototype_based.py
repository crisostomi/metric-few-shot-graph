import abc
from typing import Dict, List

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from fs_grl.data.episode.episode_batch import MolecularEpisodeBatch
from fs_grl.data.utils import SupportsAggregation
from fs_grl.modules.architectures.prototype_based import MolecularPrototypeBased
from fs_grl.modules.components.deepsets import DeepSetsEmbedder
from fs_grl.modules.components.graph_embedder import GraphEmbedder
from fs_grl.modules.components.mlp import MLP


class MolecularGNNPrototypeBased(MolecularPrototypeBased, abc.ABC):
    def __init__(
        self,
        cfg: DictConfig,
        feature_dim: int,
        num_classes: int,
        metric_scaling_factor: float,
        loss_weights: Dict,
        supports_aggregation: SupportsAggregation,
        **kwargs,
    ):
        super().__init__(loss_weights)
        self.cfg = cfg

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.embedder: GraphEmbedder = instantiate(
            self.cfg.embedder, cfg=self.cfg.embedder, feature_dim=self.feature_dim, _recursive_=False
        )

        assert supports_aggregation == "mean" or "deepsets"
        self.supports_aggregation = supports_aggregation

        if metric_scaling_factor is not None:
            self.register_parameter("metric_scaling_factor", nn.Parameter(torch.tensor(metric_scaling_factor)))
        else:
            self.metric_scaling_factor = 1.0

        if self.supports_aggregation == SupportsAggregation.DEEPSETS.value:

            phi = torch.nn.Sequential(
                MLP(
                    num_layers=2,
                    input_dim=self.embedder.embedding_dim,
                    output_dim=self.embedder.embedding_dim,
                    hidden_dim=self.embedder.embedding_dim,
                    non_linearity=torch.nn.ReLU(),
                    norm=nn.BatchNorm1d,
                ),
            )

            rho = torch.nn.Sequential(
                MLP(
                    num_layers=2,
                    input_dim=self.embedder.embedding_dim,
                    output_dim=self.embedder.embedding_dim,
                    hidden_dim=self.embedder.embedding_dim,
                    non_linearity=torch.nn.ReLU(),
                    norm=nn.BatchNorm1d,
                ),
            )
            self.deep_sets_embedder = DeepSetsEmbedder(phi=phi, rho=rho)

    def forward(self, batch: MolecularEpisodeBatch):
        """
        :param batch:
        :return
        """

        embedded_supports = self.embed_supports(batch)

        # shape (num_queries_batch, hidden_dim)
        embedded_queries = self.embed_queries(batch)

        # shape (num_classes_per_episode, hidden_dim)
        prototypes_dict = self.compute_prototypes(embedded_supports, batch)

        similarities = self.compute_queries_prototypes_correlations_batch(embedded_queries, prototypes_dict, batch)

        return {
            "embedded_queries": embedded_queries,
            "prototypes_dicts": prototypes_dict,
            "similarities": similarities,
        }

    def aggregate_supports(self, supports):
        if self.supports_aggregation == SupportsAggregation.MEAN:
            prototype = supports.mean(dim=0)
        elif self.supports_aggregation == SupportsAggregation.DEEPSETS:
            prototype = self.deep_sets_embedder(supports)
        else:
            raise NotImplementedError(f"No such aggregation {self.supports_aggregation}")
        return prototype

    @abc.abstractmethod
    def compute_queries_prototypes_correlations_batch(self, embedded_queries, class_prototypes, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_classification_loss(self, embedded_queries, class_prototypes, batch: MolecularEpisodeBatch, **kwargs):
        pass

    @property
    def embedding_dim(self):
        return self.embedder.embedding_dim


class MolecularGNNPrototypeFromGraphs(MolecularGNNPrototypeBased, abc.ABC):
    def __init__(
        self,
        cfg: DictConfig,
        feature_dim: int,
        num_classes: int,
        metric_scaling_factor: float,
        loss_weights: Dict,
        **kwargs,
    ):
        super().__init__(cfg, feature_dim, num_classes, metric_scaling_factor, loss_weights, **kwargs)

    def compute_prototypes(
        self, episode_embedded_supports: torch.Tensor, batch: MolecularEpisodeBatch, **kwargs
    ) -> List[Dict[int, torch.Tensor]]:
        """
        Computes the prototype of each class as an aggregation of the embedded supports for that class

        :param episode_embedded_supports: tensor ~ (num_supports_batch, embedding_dim)
        :param batch:
        :return a list where each entry corresponds to the class prototypes
                of an episode as a dict (Ex. all_class_prototypes[0] contains the dict
                of the class prototypes of the first episode, and so on)
        """
        num_episodes = batch.num_episodes

        # embedded supports for each episode, each has shape (num_supports_per_episode, hidden_dim)
        embedded_supports_per_episode = episode_embedded_supports.view(
            (num_episodes, batch.episode_hparams.num_supports_per_episode, self.embedding_dim)
        )

        # sequence of labels for each episode, each has shape (num_supports_per_episode)
        support_labels_by_episode = batch.supports.y.view(
            (num_episodes, batch.episode_hparams.num_supports_per_episode)
        )

        labels_per_episode = batch.active_or_not_labels.view(
            num_episodes, batch.episode_hparams.num_classes_per_episode
        )

        all_prototypes = []
        for episode in range(num_episodes):

            prototypes_episode = self.compute_episode_prototypes(
                support_labels_by_episode[episode], embedded_supports_per_episode[episode], labels_per_episode[episode]
            )

            all_prototypes.append(prototypes_episode)

        return all_prototypes

    def compute_episode_prototypes(
        self,
        episode_support_labels,
        episode_embedded_supports,
        episode_labels,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute the label prototypes for a single episode

        :param episode_support_labels: labels for the supports in the episode
        :param episode_embedded_supports: embeddings for the supports in the episode
        :param episode_labels: global labels for the episode

        :return mapping label -> corresponding prototype embedding
        """

        prototypes_episode = {}

        for label in episode_labels:
            label_prototype = self.compute_label_prototypes(label, episode_embedded_supports, episode_support_labels)
            prototypes_episode[label.item()] = label_prototype

        return prototypes_episode

    def compute_label_prototypes(
        self, label: torch.Tensor, embedded_supports: torch.Tensor, support_labels: torch.Tensor
    ) -> torch.Tensor:
        """

        :param label: label for which we compute the prototype
        :param embedded_supports: tensor (N*K, embedding_dim) containing the embedded supports in the episode
        :param support_labels: labels for the supports in the episode
        :return
        """
        device = embedded_supports.device
        num_supports = embedded_supports.shape[0]

        # pick supports having label "label"
        label_supports_indices = torch.arange(num_supports, device=device)[support_labels == label]

        # obtain the label prototype as the mean of those supports
        label_supports = embedded_supports[label_supports_indices]
        label_prototype = self.aggregate_supports(label_supports)

        return label_prototype


class MolecularGNNPrototypeFromNodes(MolecularGNNPrototypeBased, abc.ABC):
    def __init__(
        self,
        cfg: DictConfig,
        feature_dim: int,
        num_classes: int,
        metric_scaling_factor: float,
        loss_weights: Dict,
        **kwargs,
    ):
        super().__init__(cfg, feature_dim, num_classes, metric_scaling_factor, loss_weights, **kwargs)

    def compute_prototypes(
        self, episode_embedded_support_nodes: torch.Tensor, batch: MolecularEpisodeBatch, **kwargs
    ) -> List[Dict[int, torch.Tensor]]:
        """
        Computes the prototype of each class as the mean of the embedded supports for that class

        :param episode_embedded_support_nodes: tensor ~ (num_support_nodes_batch, embedding_dim)
        :param batch:
        :return a list where each entry corresponds to the class prototypes
                of an episode as a dict (Ex. all_class_prototypes[0] contains the dict
                of the class prototypes of the first episode, and so on)
        """
        num_episodes = batch.num_episodes

        graph_cumsizes_in_batch = batch.supports.ptr

        episode_delimiters = torch.tensor(
            [i * batch.episode_hparams.num_supports_per_episode for i in range(0, batch.num_episodes + 1)],
            dtype=torch.long,
        )
        episode_support_num_nodes = torch.tensor(
            [graph_cumsizes_in_batch[i] for i in episode_delimiters], dtype=torch.long
        )

        episode_support_sizes = episode_support_num_nodes[1:] - episode_support_num_nodes[:-1]

        graph_sizes = graph_cumsizes_in_batch[1:] - graph_cumsizes_in_batch[:-1]
        graph_sizes_per_episode = graph_sizes.split(
            tuple([batch.episode_hparams.num_supports_per_episode] * num_episodes)
        )

        # sequence of embedded supports for each episode, each has shape (num_support_nodes_episode, hidden_dim)
        embedded_support_nodes_per_episode = episode_embedded_support_nodes.split(tuple(episode_support_sizes))

        # sequence of labels for each episode, each has shape (num_supports_per_episode)
        support_labels_by_episode = batch.supports.y.view(
            (num_episodes, batch.episode_hparams.num_supports_per_episode)
        )
        labels_per_episode = batch.active_or_not_labels.view(
            num_episodes, batch.episode_hparams.num_classes_per_episode
        )

        all_prototypes = []
        for episode in range(num_episodes):

            episode_embedded_supports = embedded_support_nodes_per_episode[episode]
            episode_support_labels = support_labels_by_episode[episode]
            episode_labels = labels_per_episode[episode]
            episode_graph_sizes = graph_sizes_per_episode[episode]

            prototypes_episode = self.compute_episode_prototypes(
                episode_support_labels,
                episode_embedded_supports,
                episode_labels,
                episode_graph_sizes,
            )

            all_prototypes.append(prototypes_episode)

        return all_prototypes

    def compute_episode_prototypes(
        self,
        episode_support_labels,
        episode_embedded_supports,
        episode_labels,
        episode_graph_sizes,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute the label prototypes for a single episode

        :param episode_support_labels: labels for the supports in the episode
        :param episode_embedded_supports: embeddings for the supports in the episode
        :param episode_labels: global labels for the episode
        :param episode_graph_sizes:

        :return mapping label -> corresponding prototype embedding
        """

        prototypes_episode = {}

        for label in episode_labels:
            label_prototype = self.compute_label_prototypes(
                label, episode_embedded_supports, episode_support_labels, episode_graph_sizes
            )
            prototypes_episode[label.item()] = label_prototype

        return prototypes_episode

    def compute_label_prototypes(
        self,
        label: torch.Tensor,
        embedded_supports: torch.Tensor,
        support_labels: torch.Tensor,
        episode_graph_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """

        :param episode_graph_sizes:
        :param label: label for which we compute the prototype
        :param embedded_supports: tensor (num_support_nodes_episode, embedding_dim) containing the
                                    embedded support nodes in the episode
        :param support_labels: labels for the supports in the episode
        :return
        """
        device = embedded_supports.device

        num_support_nodes = embedded_supports.shape[0]

        # pick support nodes having label "label"
        node_level_labels = support_labels.repeat_interleave(episode_graph_sizes)
        label_support_nodes_indices = torch.arange(num_support_nodes, device=device)[node_level_labels == label]

        # obtain the label prototype as the mean of those supports
        label_support_nodes = embedded_supports[label_support_nodes_indices]
        label_prototype = self.aggregate_supports(label_support_nodes)

        return label_prototype
