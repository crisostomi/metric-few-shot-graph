import abc
from typing import Dict, List

import torch
from hydra.utils import instantiate

from fs_grl.data.episode import EpisodeBatch
from fs_grl.modules.baselines.prototypical_dml import PrototypicalDML
from fs_grl.modules.deepsets import DeepSetsEmbedder
from fs_grl.modules.graph_embedder import GraphEmbedder
from fs_grl.modules.mlp import MLP


class GNNEmbeddingSimilarity(PrototypicalDML, abc.ABC):
    def __init__(
        self, cfg, feature_dim, num_classes, supports_aggregation="deepsets", prototypes_from_nodes=False, **kwargs
    ):
        super().__init__()
        self.cfg = cfg

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.embedder: GraphEmbedder = instantiate(
            self.cfg.embedder, cfg=self.cfg.embedder, feature_dim=self.feature_dim, _recursive_=False
        )

        self.supports_aggregation = supports_aggregation
        self.prototypes_from_nodes = prototypes_from_nodes

        if self.supports_aggregation == "deepsets":

            phi = torch.nn.Sequential(
                MLP(
                    num_layers=2,
                    input_dim=self.embedder.embedding_dim,
                    output_dim=self.embedder.embedding_dim,
                    hidden_dim=self.embedder.embedding_dim,
                    non_linearity=torch.nn.ReLU(),
                    use_batch_norm=True,
                ),
            )

            rho = torch.nn.Sequential(
                MLP(
                    num_layers=2,
                    input_dim=self.embedder.embedding_dim,
                    output_dim=self.embedder.embedding_dim,
                    hidden_dim=self.embedder.embedding_dim,
                    non_linearity=torch.nn.ReLU(),
                    use_batch_norm=True,
                ),
            )
            self.deep_sets_embedder = DeepSetsEmbedder(phi=phi, rho=rho)

    def forward(self, batch: EpisodeBatch):
        """
        :param batch:
        :return:
        """

        graph_level = not self.prototypes_from_nodes
        embedded_supports = self.embed_supports(batch, graph_level=graph_level)

        # shape (num_queries_batch, hidden_dim)
        embedded_queries = self.embed_queries(batch)

        # shape (num_classes_per_episode, hidden_dim)
        class_prototypes = self.get_prototypes(embedded_supports, batch)

        similarities = self.get_queries_prototypes_similarities_batch(embedded_queries, class_prototypes, batch)

        return {
            "embedded_queries": embedded_queries,
            "class_prototypes": class_prototypes,
            "similarities": similarities,
        }

    def get_prototypes(self, episode_embedded_supports: torch.Tensor, batch: EpisodeBatch):
        if self.prototypes_from_nodes:
            return self.get_prototypes_from_nodes(episode_embedded_supports, batch)
        else:
            return self.get_prototypes_from_graphs(episode_embedded_supports, batch)

    def get_prototypes_from_graphs(
        self, episode_embedded_supports: torch.Tensor, batch: EpisodeBatch
    ) -> List[Dict[int, torch.Tensor]]:
        """
        Computes the prototype of each class as the mean of the embedded supports for that class

        :param episode_embedded_supports: tensor ~ (num_supports_batch, embedding_dim)
        :param batch:
        :return: a list where each entry corresponds to the class prototypes of an episode as a dict (Ex. all_class_prototypes[0]
                 contains the dict of the class prototypes of the first episode, and so on)
        """
        num_episodes = batch.num_episodes

        # sequence of embedded supports for each episode, each has shape (num_supports_per_episode, hidden_dim)
        embedded_supports_per_episode = episode_embedded_supports.split(
            tuple([batch.num_supports_per_episode] * num_episodes)
        )

        # sequence of labels for each episode, each has shape (num_supports_per_episode)
        support_labels_by_episode = batch.supports.y.split(tuple([batch.num_supports_per_episode] * num_episodes))
        labels_per_episode = batch.global_labels.split([batch.episode_hparams.num_classes_per_episode] * num_episodes)

        all_prototypes = []
        for episode in range(num_episodes):

            episode_embedded_supports = embedded_supports_per_episode[episode]
            episode_support_labels = support_labels_by_episode[episode]
            episode_labels = labels_per_episode[episode]

            prototypes_episode = self.compute_episode_prototypes(
                episode_support_labels, episode_embedded_supports, episode_labels
            )

            all_prototypes.append(prototypes_episode)

        return all_prototypes

    def get_prototypes_from_nodes(
        self, episode_embedded_support_nodes: torch.Tensor, batch: EpisodeBatch
    ) -> List[Dict[int, torch.Tensor]]:
        """
        Computes the prototype of each class as the mean of the embedded supports for that class

        :param episode_embedded_support_nodes: tensor ~ (num_support_nodes_batch, embedding_dim)
        :param batch:
        :return: a list where each entry corresponds to the class prototypes of an episode as a dict (Ex. all_class_prototypes[0]
                 contains the dict of the class prototypes of the first episode, and so on)
        """
        num_episodes = batch.num_episodes

        graph_cumsizes_in_batch = batch.supports.ptr

        episode_delimiters = torch.tensor(
            [i * batch.num_supports_per_episode for i in range(0, batch.num_episodes + 1)], dtype=torch.long
        )
        episode_support_num_nodes = torch.tensor(
            [graph_cumsizes_in_batch[i] for i in episode_delimiters], dtype=torch.long
        )

        episode_support_sizes = episode_support_num_nodes[1:] - episode_support_num_nodes[:-1]

        graph_sizes = graph_cumsizes_in_batch[1:] - graph_cumsizes_in_batch[:-1]
        graph_sizes_per_episode = graph_sizes.split(tuple([batch.num_supports_per_episode] * num_episodes))

        # sequence of embedded supports for each episode, each has shape (num_support_nodes_episode, hidden_dim)
        embedded_support_nodes_per_episode = episode_embedded_support_nodes.split(tuple(episode_support_sizes))

        # sequence of labels for each episode, each has shape (num_supports_per_episode)
        support_labels_by_episode = batch.supports.y.split(tuple([batch.num_supports_per_episode] * num_episodes))
        labels_per_episode = batch.global_labels.split([batch.episode_hparams.num_classes_per_episode] * num_episodes)

        all_prototypes = []
        for episode in range(num_episodes):

            episode_embedded_supports = embedded_support_nodes_per_episode[episode]
            episode_support_labels = support_labels_by_episode[episode]
            episode_labels = labels_per_episode[episode]
            episode_graph_sizes = graph_sizes_per_episode[episode]

            prototypes_episode = self.compute_episode_prototypes(
                episode_support_labels, episode_embedded_supports, episode_labels, episode_graph_sizes, from_nodes=True
            )

            all_prototypes.append(prototypes_episode)

        return all_prototypes

    def compute_episode_prototypes(
        self,
        episode_support_labels,
        episode_embedded_supports,
        episode_labels,
        episode_graph_sizes=None,
        from_nodes=False,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute the label prototypes for a single episode

        :param episode_support_labels: labels for the supports in the episode
        :param episode_embedded_supports: embeddings for the supports in the episode
        :param episode_labels: global labels for the episode

        :return: mapping label -> corresponding prototype embedding
        """

        prototypes_episode = {}

        for label in episode_labels:
            if from_nodes:
                label_prototype = self.compute_label_prototype_from_nodes(
                    label, episode_embedded_supports, episode_support_labels, episode_graph_sizes
                )
            else:
                label_prototype = self.compute_label_prototype_from_graphs(
                    label, episode_embedded_supports, episode_support_labels
                )
            prototypes_episode[label.item()] = label_prototype

        return prototypes_episode

    def compute_label_prototype_from_graphs(
        self, label: torch.Tensor, embedded_supports: torch.Tensor, support_labels: torch.Tensor
    ) -> torch.Tensor:
        """

        :param label: label for which we compute the prototype
        :param embedded_supports: tensor (N*K, embedding_dim) containing the embedded supports in the episode
        :param support_labels: labels for the supports in the episode
        :return:
        """
        device = embedded_supports.device
        num_supports = embedded_supports.shape[0]

        # pick supports having label "label"
        label_supports_indices = torch.arange(num_supports, device=device)[support_labels == label]

        # obtain the label prototype as the mean of those supports
        label_supports = embedded_supports[label_supports_indices]
        label_prototype = self.aggregate_supports(label_supports)

        return label_prototype

    def compute_label_prototype_from_nodes(
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
        :return:
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

    def aggregate_supports(self, supports):
        if self.supports_aggregation == "mean":
            prototype = supports.mean(dim=0)
        elif self.supports_aggregation == "deepsets":
            prototype = self.deep_sets_embedder(supports)
        else:
            raise NotImplementedError(f"No such aggregation {self.supports_aggregation}")
        return prototype

    @abc.abstractmethod
    def get_queries_prototypes_similarities_batch(self, embedded_queries, class_prototypes, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, embedded_queries, class_prototypes, batch: EpisodeBatch, **kwargs):
        pass
