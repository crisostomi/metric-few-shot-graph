import math
import random
from abc import ABC
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from fs_grl.data.episode.episode import Episode, EpisodeHParams
from fs_grl.data.utils import get_label_to_samples_map


class EpisodicDataset(ABC):
    def __init__(
        self,
        num_episodes: int,
        samples: List[Data],
        stage_labels: List[int],
        episode_hparams: EpisodeHParams,
    ):
        """

        :param num_episodes: number of epochs per epoch
        :param samples: list of graph samples
        :param stage_labels: labels of the current stage (i.e. base, val and novel)
        :param episode_hparams: number of classes N in the episode, number of supports per class K
                                number of queries per class Q
        """
        super().__init__()
        self.num_episodes = num_episodes
        self.episode_hparams = episode_hparams

        self.samples = samples

        self.stage_labels = stage_labels

        self.samples_by_label: Dict[int : List[Data]] = get_label_to_samples_map(self.samples)

    def sample_episode(self):
        f"""
        Creates an episode by first sampling {self.episode_hparams.num_classes_per_episode} classes
        and then sampling K supports and Q queries for each class
        :return
        """

        episode_labels = self.sample_labels()

        supports: List[Data] = []
        queries: List[Data] = []

        for label in episode_labels:
            label_supports_queries = self.sample_label_queries_supports(label)

            supports += label_supports_queries["supports"]
            queries += label_supports_queries["queries"]

        random.shuffle(supports)
        random.shuffle(queries)

        return Episode(supports, queries, episode_labels, episode_hparams=self.episode_hparams)

    def sample_label_queries_supports(self, label: int) -> Dict[str, List[Data]]:
        f"""
        Given a label {label}, samples K support and Q queries. These are always disjoint inside the episode.
        """

        all_label_supports: List[Data] = self.samples_by_label[label]

        label_samples_episode = random.sample(
            all_label_supports,
            self.episode_hparams.num_supports_per_class + self.episode_hparams.num_queries_per_class,
        )

        label_supports_episode = label_samples_episode[: self.episode_hparams.num_supports_per_class]
        label_queries_episode = label_samples_episode[self.episode_hparams.num_supports_per_class :]

        return {"supports": label_supports_episode, "queries": label_queries_episode}

    def sample_labels(self) -> List:
        """
        Sample N labels for an episode from the stage labels

        :return
        """
        sampled_labels = random.sample(self.stage_labels, self.episode_hparams.num_classes_per_episode)
        return sorted(sampled_labels)


class IterableEpisodicDataset(torch.utils.data.IterableDataset, EpisodicDataset):
    def __init__(
        self,
        num_episodes: int,
        samples: List[Data],
        stage_labels: List[int],
        episode_hparams: EpisodeHParams,
    ):
        super().__init__(
            num_episodes=num_episodes,
            samples=samples,
            stage_labels=stage_labels,
            episode_hparams=episode_hparams,
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            per_worker = self.num_episodes
            worker_id = 0
        else:  # in a worker process
            worker_id = worker_info.id
            per_worker = math.ceil(int(self.num_episodes / float(worker_info.num_workers)))

        random.seed(worker_id)

        return iter(self.sample_episode() for _ in range(per_worker))

    def __getitem__(self, index):
        raise NotImplementedError


class MapEpisodicDataset(Dataset, EpisodicDataset):
    def __init__(
        self,
        num_episodes: int,
        samples: List[Data],
        stage_labels: List[int],
        episode_hparams: EpisodeHParams,
    ):
        super().__init__(
            num_episodes=num_episodes,
            samples=samples,
            stage_labels=stage_labels,
            episode_hparams=episode_hparams,
        )

        self.episodes = [self.sample_episode() for _ in range(self.num_episodes)]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]
