import itertools
from typing import Dict, List

import torch
from torch_geometric.data import Batch, Data

from fs_grl.data.episode.episode import Episode, EpisodeHParams
from fs_grl.data.utils import SampleType, flatten


class EpisodeBatch:
    def __init__(
        self,
        supports: Batch,
        queries: Batch,
        global_labels: torch.Tensor,
        episode_hparams: EpisodeHParams,
        num_episodes: int,
    ):
        """

        :param supports: supports for all the episodes in the batch (BxNxK)
        :param queries: queries for all the episodes in the batch (BxNxQ)
        :param global_labels: tensor containing for each episode the considered global labels (BxN)
        :param episode_hparams: N, K, Q
        :param num_episodes: how many episodes per batch, i.e. batch size
        """
        assert global_labels.shape[0] == num_episodes * episode_hparams.num_classes_per_episode

        self.episode_hparams = episode_hparams
        self.num_episodes = num_episodes

        self.supports = supports
        self.queries = queries
        self.global_labels = global_labels

        self.num_samples_per_episode = {
            SampleType.QUERY: self.episode_hparams.num_queries_per_episode,
            SampleType.SUPPORT: self.episode_hparams.num_supports_per_episode,
        }
        self.samples = {SampleType.QUERY: self.queries, SampleType.SUPPORT: self.supports}

    @classmethod
    def episode_batch_kwargs(cls, episode_list: List[Episode], episode_hparams: EpisodeHParams) -> Dict:
        """
        Returns the arguments for the EpisodeBatch constructor

        :param episode_list: list of episodes
        :param episode_hparams: N, K and Q

        :return:
        """

        # B * N * K
        supports: List[Data] = flatten([episode.supports for episode in episode_list])
        # B * N * Q
        queries: List[Data] = flatten([episode.queries for episode in episode_list])
        # B * N
        global_labels: List[int] = flatten([episode.global_labels for episode in episode_list])

        supports_batch: Batch = Batch.from_data_list(supports)
        queries_batch: Batch = Batch.from_data_list(queries)
        global_labels_batch = torch.tensor(global_labels)

        return {
            "supports": supports_batch,
            "queries": queries_batch,
            "global_labels": global_labels_batch,
            "episode_hparams": episode_hparams,
            "num_episodes": len(episode_list),
        }

    @classmethod
    def from_episode_list(
        cls,
        episode_list: List[Episode],
        episode_hparams: EpisodeHParams,
    ) -> "EpisodeBatch":
        """
        Constructs a new EpisodeBatch.

        :param episode_list: list of episodes
        :param episode_hparams: N, K and Q

        :return: new EpisodeBatch
        """

        episode_batch_arguments = cls.episode_batch_kwargs(episode_list, episode_hparams)
        episode_batch = cls(**episode_batch_arguments)

        return episode_batch

    @property
    def feature_dim(self):
        assert self.supports.x.shape[-1] == self.queries.x.shape[-1]
        return self.supports.x.shape[-1]

    def split_in_episodes(self, sample_type: SampleType):
        """
        Splits the EpisodeBatch into episodes

        EpisodeBatch contains BxNxK support graphs and BxNxQ query graphs

        Returns B lists of (NxK) graphs each if sample_type is support
            or  B lists of (NxQ) graphs each if sample_type is query
        """

        samples_batch = self.samples[sample_type]

        # list containing (BxNxK) supports or (BxNxQ) queries
        samples_list = Batch.to_data_list(samples_batch)

        # how many supports or queries for episode
        num_samples_ep = self.num_samples_per_episode[sample_type]

        samples_by_episode = [
            samples_list[i * num_samples_ep : i * num_samples_ep + num_samples_ep] for i in range(self.num_episodes)
        ]

        return samples_by_episode

    def get_global_labels_by_episode(self) -> torch.Tensor:
        """
        Split global labels tensor ~ (B*N) by episode

        :return: tensor (BxN)
        """

        global_labels_by_episode = self.global_labels.view(
            (self.num_episodes, self.episode_hparams.num_classes_per_episode)
        )
        return global_labels_by_episode

    def get_support_labels_by_episode(self) -> torch.Tensor:
        """
        Split support labels tensor ~ (B*N*K) by episode

        :return: tensor (B, N*K)
        """
        support_labels_by_episode = self.supports.y.view(
            (self.num_episodes, self.episode_hparams.num_supports_per_episode)
        )
        return support_labels_by_episode

    def get_query_labels_by_episode(self) -> torch.Tensor:
        """
        Split query labels tensor ~ (B*N*Q) by episode

        :return: tensor (B, N*Q)
        """
        query_labels_by_episode = self.queries.y.view((self.num_episodes, self.episode_hparams.num_queries_per_episode))
        return query_labels_by_episode

    def to(self, device):
        self.supports = self.supports.to(device)
        self.queries = self.queries.to(device)
        self.global_labels = self.global_labels.to(device)

    def pin_memory(self):
        for key, attr in self.__dict__.items():
            if attr is not None and hasattr(attr, "pin_memory"):
                attr.pin_memory()

        return self


class CosineEpisodeBatch(EpisodeBatch):
    def __init__(
        self,
        supports: Batch,
        queries: Batch,
        global_labels: torch.Tensor,
        episode_hparams: EpisodeHParams,
        num_episodes: int,
        cosine_targets: torch.Tensor,
    ):
        """

        :param supports: supports for all the episodes in the batch (BxNxK)
        :param queries: queries for all the episodes in the batch (BxNxQ)
        :param global_labels: tensor containing for each episode the considered global labels (BxN)
        :param episode_hparams: N, K, Q
        :param num_episodes: how many episodes per batch, i.e. batch size
        :param cosine_targets:
        """
        super().__init__(supports, queries, global_labels, episode_hparams, num_episodes)

        self.cosine_targets = cosine_targets

    @classmethod
    def from_episode_list(
        cls,
        episode_list: List[Episode],
        episode_hparams: EpisodeHParams,
    ) -> "CosineEpisodeBatch":

        kwargs = super().episode_batch_kwargs(episode_list, episode_hparams)

        # shape (B*(N*Q)*N)
        cosine_targets = cls.get_cosine_targets(episode_list)

        kwargs["cosine_targets"] = cosine_targets
        episode_batch = cls(**kwargs)

        return episode_batch

    @classmethod
    def get_cosine_targets(cls, episode_list: List[Episode]) -> torch.Tensor:
        """
        :param episode_list: list of episodes in the batch

        :return tensor ~(B*(N*Q)*N) where for each episode in [1, ..., B] there are all the
                 target similarities between the N*Q queries and the N considered global labels
                 Query q in [1, ..., (N*Q)] and label l in [1, ..., N] will have sim(q, l) = 1 if
                 query q has label l, else -1
        """
        cosine_targets = []
        for episode in episode_list:

            # shape ((N*Q)*N)
            episode_cosine_targets = []

            for query, label in itertools.product(episode.queries, episode.global_labels):
                query_label_similarity = (query.y.item() == label) * 2 - 1
                episode_cosine_targets.append(query_label_similarity)

            cosine_targets.append(torch.tensor(episode_cosine_targets, dtype=torch.long))

        cosine_targets = torch.cat(cosine_targets, dim=-1)

        return cosine_targets

    def to(self, device):
        super().to(device)
        self.cosine_targets = self.cosine_targets.to(device)
