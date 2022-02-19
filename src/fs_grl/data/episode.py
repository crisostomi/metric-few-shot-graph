import itertools
from dataclasses import dataclass
from typing import List

import torch
from torch_geometric.data import Batch, Data


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


class Episode:
    def __init__(self, supports: List[Data], queries: List[Data], labels, episode_hparams: EpisodeHParams):
        """
        N classes, K samples each, Q queries each
        :param supports: shape (N*K), contains K support samples for each class
        :param queries: shape (N*Q), contains Q queries for each class
        """
        self.supports = supports
        self.queries = queries
        self.labels = labels

        self.episode_hparams = episode_hparams


class EpisodeBatch(Episode):
    def __init__(
        self,
        supports,
        queries,
        labels,
        episode_hparams,
        supports_len,
        queries_len,
        num_episodes,
        cosine_targets: torch.Tensor,
        label_targets: torch.Tensor,
    ):
        super().__init__(supports, queries, labels, episode_hparams)
        self.supports_len = supports_len
        self.queries_len = queries_len
        self.num_episodes = num_episodes
        self.cosine_targets = cosine_targets
        self.label_targets = label_targets

        self.batch_size = len(self.supports) / (
            self.episode_hparams.num_supports_per_class * self.episode_hparams.num_classes_per_episode
        )

    @classmethod
    def from_episode_list(cls, episode_list: List[Episode], episode_hparams):

        # C * K * batch_size
        supports: List[Data] = [x for episode in episode_list for x in episode.supports]
        # C * Q * batch_size
        queries: List[Data] = [x for episode in episode_list for x in episode.queries]
        labels: List = [x for episode in episode_list for x in episode.labels]

        supports_batch = Batch.from_data_list(supports)
        queries_batch = Batch.from_data_list(queries)
        labels_batch = torch.tensor(labels)

        batch_size = len(episode_list)

        supports_len = torch.tensor(
            [sum(x.num_nodes for x in episode.supports) for episode in episode_list], dtype=torch.long
        )
        queries_len = torch.tensor(
            [sum(x.num_nodes for x in episode.queries) for episode in episode_list], dtype=torch.long
        )

        cosine_targets = torch.cat(
            [
                # TODO: check consistency here
                ((query.y == label) * 2 - 1).long()
                for episode in episode_list
                for query, label in itertools.product(episode.queries, episode.labels)
            ],
            dim=-1,
        )

        targets = cosine_targets.reshape(-1, episode_hparams.num_classes_per_episode)
        targets = targets.argmax(dim=-1)

        return cls(
            supports=supports_batch,
            queries=queries_batch,
            labels=labels_batch,
            episode_hparams=episode_hparams,
            supports_len=supports_len,
            queries_len=queries_len,
            num_episodes=batch_size,
            cosine_targets=cosine_targets,
            label_targets=targets,
        )

    def to(self, device):
        # TODO: check
        self.supports = self.supports.to(device)
        self.queries = self.queries.to(device)
        self.cosine_targets = self.cosine_targets.to(device)
        self.label_targets = self.label_targets.to(device)
        self.labels = self.labels.to(device)

    def pin_memory(self):
        for key, attr in self.__dict__.items():
            if attr is not None and hasattr(attr, "pin_memory"):
                attr.pin_memory()

        return self
