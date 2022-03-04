import itertools
from dataclasses import dataclass
from typing import List, Union

import torch
from torch_geometric.data import Batch, Data

from fs_grl.data.utils import flatten


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


class Episode:
    def __init__(
        self,
        supports: Union[List[Data], Batch],
        queries: Union[List[Data], Batch],
        global_labels: List[int],
        episode_hparams: EpisodeHParams,
    ):
        """
        N classes, K samples each, Q queries each

        :param supports: shape (N*K), contains K support samples for each class
        :param queries: shape (N*Q), contains Q queries for each class
        :param global_labels: subset of the stage labels sampled for the episode
        :param episode_hparams: N, K and Q
        """
        self.supports = supports
        self.queries = queries
        self.global_labels = global_labels

        self.episode_hparams = episode_hparams
        self.num_queries_per_episode = (
            self.episode_hparams.num_queries_per_class * self.episode_hparams.num_classes_per_episode
        )
        self.num_supports_per_episode = (
            self.episode_hparams.num_supports_per_class * self.episode_hparams.num_classes_per_episode
        )


class EpisodeBatch(Episode):
    def __init__(
        self,
        supports: Batch,
        queries: Batch,
        global_labels: torch.Tensor,
        episode_hparams: EpisodeHParams,
        num_episodes: int,
        cosine_targets: torch.Tensor,
        local_labels: torch.Tensor,
    ):
        """

        :param supports: supports for all the episodes in the batch
        :param queries: queries for all the episodes in the batch
        :param global_labels: tensor containing for each episode the considered global labels
        :param episode_hparams: N, K, Q
        :param num_episodes: how many episodes per batch
        :param cosine_targets:
        :param local_labels: labels remapped to 0,..,N-1 which are local to the single episodes
                             i.e. they are not consistent through the batch
        """
        super().__init__(
            supports=supports, queries=queries, global_labels=global_labels, episode_hparams=episode_hparams
        )

        self.num_episodes = num_episodes
        self.cosine_targets = cosine_targets
        self.local_labels = local_labels

    @classmethod
    def from_episode_list(cls, episode_list: List[Episode], episode_hparams: EpisodeHParams) -> "EpisodeBatch":

        # N * K * batch_size
        supports: List[Data] = flatten([episode.supports for episode in episode_list])
        # N * Q * batch_size
        queries: List[Data] = flatten([episode.queries for episode in episode_list])
        # N * batch_size
        global_labels: List[int] = flatten([episode.global_labels for episode in episode_list])

        supports_batch: Batch = Batch.from_data_list(supports)
        queries_batch: Batch = Batch.from_data_list(queries)
        global_labels_batch = torch.tensor(global_labels)

        num_episodes = len(episode_list)

        # shape (B*N*Q*N)
        cosine_targets = cls.get_cosine_targets(episode_list)

        # shape (N*Q*B, N)
        local_labels = cosine_targets.reshape(-1, episode_hparams.num_classes_per_episode)
        local_labels = local_labels.argmax(dim=-1)

        return cls(
            supports=supports_batch,
            queries=queries_batch,
            global_labels=global_labels_batch,
            episode_hparams=episode_hparams,
            num_episodes=num_episodes,
            cosine_targets=cosine_targets,
            local_labels=local_labels,
        )

    @classmethod
    def get_cosine_targets(cls, episode_list: List[Episode]) -> torch.Tensor:
        """
        :param episode_list: list of episodes in the batch
        :return: tensor ~(B*(N*Q)*N) where for each episode in [1, .., B] there are all the
                 target similarities between the N*Q queries and the N considered global labels
                 Query q in [1, .., (N*Q)] and label l in [1, .., N] will have sim(q, l) = 1 if
                 query q has label l, else -1
        """
        cosine_targets = []
        for episode in episode_list:

            # shape (N*Q*N)
            episode_cosine_targets = []

            for query, label in itertools.product(episode.queries, episode.global_labels):
                query_label_similarity = (query.y.item() == label) * 2 - 1
                episode_cosine_targets.append(query_label_similarity)

            cosine_targets.append(torch.tensor(episode_cosine_targets, dtype=torch.long))

        cosine_targets = torch.cat(cosine_targets, dim=-1)

        return cosine_targets

    def to(self, device):
        self.supports = self.supports.to(device)
        self.queries = self.queries.to(device)
        self.cosine_targets = self.cosine_targets.to(device)
        self.local_labels = self.local_labels.to(device)
        self.global_labels = self.global_labels.to(device)

    def pin_memory(self):
        for key, attr in self.__dict__.items():
            if attr is not None and hasattr(attr, "pin_memory"):
                attr.pin_memory()

        return self
