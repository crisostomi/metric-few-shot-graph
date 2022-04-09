import abc
from typing import Dict, List

import torch
from hydra.utils import instantiate

from fs_grl.data.episode import EpisodeBatch
from fs_grl.modules.baselines.prototypical_dml import PrototypicalDML


class GNNEmbeddingSimilarity(PrototypicalDML, abc.ABC):
    def __init__(self, cfg, feature_dim, num_classes, **kwargs):
        super().__init__()
        self.cfg = cfg

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.embedder = instantiate(
            self.cfg.embedder, cfg=self.cfg.embedder, feature_dim=self.feature_dim, _recursive_=False
        )

    def forward(self, batch: EpisodeBatch):
        """
        :param batch:
        :return:
        """

        # shape (num_supports_batch, hidden_dim)
        embedded_supports = self.embed_supports(batch)

        # shape (num_queries_batch, hidden_dim)
        embedded_queries = self.embed_queries(batch)

        # shape (num_classes_per_episode, hidden_dim)
        class_prototypes = self.get_prototypes(embedded_supports, batch)

        similarities = self.get_queries_prototypes_similarities_batch(embedded_queries, class_prototypes, batch)

        return {
            "embedded_queries": embedded_queries,
            "embedded_supports": embedded_supports,
            "class_prototypes": class_prototypes,
            "similarities": similarities,
        }

    def get_prototypes(self, embedded_supports: torch.Tensor, batch: EpisodeBatch) -> List[Dict[int, torch.Tensor]]:
        """
        Computes the prototype of each class as the mean of the embedded supports for that class

        :param embedded_supports: tensor ~ (num_supports_batch, embedding_dim)
        :param batch:
        :return: a list where each entry corresponds to the class prototypes of an episode as a dict (Ex. all_class_prototypes[0]
                 contains the dict of the class prototypes of the first episode, and so on)
        """
        device = embedded_supports.device
        num_episodes = batch.num_episodes

        # sequence of embedded supports for each episode, each has shape (num_supports_per_episode, hidden_dim)
        embedded_supports_per_episode = embedded_supports.split(tuple([batch.num_supports_per_episode] * num_episodes))
        # sequence of labels for each episode, each has shape (num_supports_per_episode)
        support_labels_by_episode = batch.supports.y.split(tuple([batch.num_supports_per_episode] * num_episodes))
        classes_per_episode = batch.global_labels.split([batch.episode_hparams.num_classes_per_episode] * num_episodes)

        all_class_prototypes = []
        for episode in range(num_episodes):

            embedded_supports = embedded_supports_per_episode[episode]
            labels = support_labels_by_episode[episode]
            classes = classes_per_episode[episode]

            class_prototypes_episode = {}
            for cls in classes:
                class_indices = torch.arange(len(labels), device=device)[labels == cls]
                class_supports = torch.index_select(embedded_supports, dim=0, index=class_indices)
                class_prototypes = class_supports.mean(dim=0)
                class_prototypes_episode[cls.item()] = class_prototypes

            all_class_prototypes.append(class_prototypes_episode)

        return all_class_prototypes

    @abc.abstractmethod
    def get_queries_prototypes_similarities_batch(self, embedded_queries, class_prototypes, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, embedded_queries, class_prototypes, batch: EpisodeBatch, **kwargs):
        pass
