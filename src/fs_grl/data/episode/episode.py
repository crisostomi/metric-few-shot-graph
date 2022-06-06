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

        self.global_labels = global_labels
        self.supports = Episode.add_local_labels(supports)
        self.queries = Episode.add_local_labels(queries)

    @staticmethod
    def add_local_labels(samples: List[Data]):
        """
        Return samples enriched with their local label in the episode.

        Example: If the episode has N=2 labels = [7, 10], local labels will be [0, 1]
                 all the samples having label 7 will be mapped to 0,
                 and those having label 10 will be mapped to 1
        """

        new_samples = []

        global_labels = torch.stack([sample.y for sample in samples])

        # unique sorts the global labels and assigns them progressive integers correspondingly
        # e.g. [10, 7, 7, 10] --> [7, 7, 10, 10] --> [0, 0, 1, 1] --> [1, 0, 0, 1]
        local_labels = torch.unique(global_labels, return_inverse=True, sorted=True)[1]

        # only the local label will change for the same sample in different episodes,
        # remaining attributes may be shared
        for sample, local_label in zip(samples, local_labels):
            new_sample = Data(
                x=sample.x,
                edge_index=sample.edge_index,
                y=sample.y,
                local_y=local_label,
            )
            new_samples.append(new_sample)

        return new_samples
