from dataclasses import dataclass
from typing import List

import torch
from torch_geometric.data import Data


@dataclass
class EpisodeHParams:
    """
    num_supports_per_class: K, how many labeled samples there are for each class when doing few-shot
    num_queries_per_class: Q, how many samples must be predicted for each class when doing few-shot
    num_classes_per_episode: N, how many classes are considered in one few-shot episode
    """

    num_classes_per_episode: int
    num_supports_per_class: int
    num_queries_per_class: int

    def as_dict(self):
        return {
            "num_classes_per_episode": self.num_classes_per_episode,
            "num_supports_per_class": self.num_supports_per_class,
            "num_queries_per_class": self.num_queries_per_class,
        }

    @property
    def num_queries_per_episode(self):
        return self.num_queries_per_class * self.num_classes_per_episode

    @property
    def num_supports_per_episode(self):
        return self.num_supports_per_class * self.num_classes_per_episode


class Episode:
    def __init__(
        self,
        supports: List[Data],
        queries: List[Data],
        global_labels: List[int],
        episode_hparams: EpisodeHParams,
    ):
        """
        N classes, K samples each, Q queries each

        :param supports: shape (N*K), contains K support samples for each class
        :param queries: shape (N*Q), contains Q queries for each class
        :param global_labels: subset of N stage labels sampled for the episode
        :param episode_hparams: N, K and Q
        """

        self.episode_hparams = episode_hparams

        self.supports = supports
        self.queries = queries

        self.global_labels = global_labels
        Episode.add_local_labels(self.supports)
        Episode.add_local_labels(self.queries)

        support_global_labels = torch.stack([support.y for support in supports])
        query_global_labels = torch.stack([query.y for query in queries])

        self.support_local_labels = torch.unique(support_global_labels, return_inverse=True)[1]
        self.query_local_labels = torch.unique(query_global_labels, return_inverse=True)[1]

    @staticmethod
    def add_local_labels(samples: List[Data]):
        """
        In place function that adds to each sample its local label in the episode.

        Example: If the episode has N=2 labels = [7, 10], local labels will be [0, 1]
                 all the samples having label 7 will be mapped to 0,
                 and those having label 10 will be mapped to 1
        """

        global_labels = torch.stack([sample.y for sample in samples])

        # unique sorts the global labels and assigns them progressive integers correspondingly
        # e.g. [10, 7, 7, 10] --> [7, 7, 10, 10] --> [0, 0, 1, 1] --> [1, 0, 0, 1]
        local_labels = torch.unique(global_labels, return_inverse=True, sorted=True)[1]

        for ind, sample in enumerate(samples):
            sample.local_y = local_labels[ind]
