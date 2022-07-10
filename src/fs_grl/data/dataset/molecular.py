import math
import random
from abc import ABC
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from fs_grl.data.episode.episode import EpisodeHParams, MolecularEpisode


class MolecularDataset(ABC):
    def __init__(
        self,
        num_episodes: int,
        samples_by_property: Dict[int, Dict[str, List[Data]]],
        stage_properties: List[int],
        episode_hparams: EpisodeHParams,
    ):
        """

        :param num_episodes: number of epochs per epoch
        :param samples_by_property: mapping property -> list of samples for which it is active and non active
        :param stage_properties: properties of the current stage (i.e. base, val and novel)
        :param episode_hparams: number of supports per class K, number of queries per class Q
        """
        super().__init__()
        self.num_episodes = num_episodes
        self.episode_hparams = episode_hparams

        self.samples_by_property = samples_by_property

        self.stage_properties = stage_properties

    def sample_episode(self):
        """
        Creates an episode by first sampling a property and then sampling K supports and Q queries
        where it's active and non-active

        :return
        """

        episode_property = self.sample_property()

        supports: List[Data] = []
        queries: List[Data] = []

        for modality in {"positive", "negative"}:
            modality_supports_queries = self.sample_property_supports_queries(episode_property, modality)

            supports += modality_supports_queries["supports"]
            queries += modality_supports_queries["queries"]

        random.shuffle(supports)
        random.shuffle(queries)

        return MolecularEpisode(supports, queries, episode_property, episode_hparams=self.episode_hparams)

    def sample_property_supports_queries(self, property: int, modality: str) -> Dict[str, List[Data]]:
        f"""
        Given a property {property} and whether it's active or non active,
        samples K correspondingsupport and Q queries.
        """

        all_relevant_samples: List[Data] = self.samples_by_property[property][modality]

        label_samples_episode = random.sample(
            all_relevant_samples,
            self.episode_hparams.num_supports_per_class + self.episode_hparams.num_queries_per_class,
        )

        label_supports_episode = label_samples_episode[: self.episode_hparams.num_supports_per_class]
        label_queries_episode = label_samples_episode[self.episode_hparams.num_supports_per_class :]

        return {"supports": label_supports_episode, "queries": label_queries_episode}

    def sample_property(self) -> int:
        """
        Sample a property for the episode episode from the stage properties

        :return
        """
        sampled_property = random.sample(self.stage_properties, k=1)[0]
        return sampled_property


class IterableMolecularDataset(torch.utils.data.IterableDataset, MolecularDataset):
    def __init__(
        self,
        num_episodes: int,
        samples_by_property: Dict[int, Dict[str, List[Data]]],
        stage_properties: List[int],
        episode_hparams: EpisodeHParams,
    ):
        super().__init__(
            num_episodes=num_episodes,
            samples_by_property=samples_by_property,
            stage_properties=stage_properties,
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


class MapMolecularDataset(Dataset, MolecularDataset):
    def __init__(
        self,
        num_episodes: int,
        samples_by_property: Dict[int, Dict[str, List[Data]]],
        stage_properties: List[int],
        episode_hparams: EpisodeHParams,
    ):
        super().__init__(
            num_episodes=num_episodes,
            samples_by_property=samples_by_property,
            stage_properties=stage_properties,
            episode_hparams=episode_hparams,
        )

        self.episodes = [self.sample_episode() for _ in range(self.num_episodes)]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]
