import math
import random
from abc import ABC
from functools import partial
from typing import Dict, List, Union

import hydra
import omegaconf
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from torch_geometric.data import Data

from nn_core.common import PROJECT_ROOT

from fs_grl.data.episode import Episode, EpisodeBatch, EpisodeHParams
from fs_grl.data.utils import get_label_to_samples_map


class TransferSourceDataset(Dataset):
    """
    Vanilla graph dataset. Used in the base training phase of transfer learning.
    """

    def __init__(self, samples: List[Data]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class EpisodicDataset(ABC):
    def __init__(
        self,
        num_episodes: int,
        samples: Union[List, Dict],
        stage_labels: List,
        class_to_label_dict: Dict,
        episode_hparams: EpisodeHParams,
        separated_query_support: bool,
    ):
        """

        :param num_episodes:
        :param samples:
        """
        super().__init__()
        self.num_episodes = num_episodes
        self.separated_query_support = separated_query_support
        self.episode_hparams = episode_hparams

        if self.separated_query_support:
            self.supports, self.queries = samples["supports"], samples["queries"]
        else:
            self.supports, self.queries = samples, None

        self.stage_labels = stage_labels
        self.class_to_label_dict = class_to_label_dict

        self.cls_to_supports: Dict[int : List[Data]] = get_label_to_samples_map(self.supports)

        self.cls_to_queries: Dict[int : List[Data]] = (
            get_label_to_samples_map(self.queries) if self.separated_query_support else None
        )

    def sample_episode(self):
        f"""
        Creates an episode by first sampling {self.episode_hparams.num_classes_per_episode} classes
        and then sampling K supports and Q queries for each class
        :return:
        """
        labels = sorted(random.sample(self.stage_labels, self.episode_hparams.num_classes_per_episode))

        supports = []
        queries = []

        for label in labels:
            label_supports_queries = self.sample_label_queries_supports(label)

            supports += label_supports_queries["supports"]
            queries += label_supports_queries["queries"]

        random.shuffle(supports)
        random.shuffle(queries)

        return Episode(supports, queries, labels, episode_hparams=self.episode_hparams)

    def sample_label_queries_supports(self, label: int) -> Dict[str, List[Data]]:
        f"""
        Given a label, samples K support and Q queries. These are always disjoint inside the episode,
        but the same sample may be query in one episode and support in another if {self.separated_query_support}
        is false. If it is true instead, the sets are disjoint even in different episodes.
        """
        all_cls_supports: List[Data] = self.cls_to_supports[label]

        if self.separated_query_support:
            cls_supports = random.sample(all_cls_supports, self.episode_hparams.num_supports_per_class)

            all_cls_queries: List[Data] = self.cls_to_queries[label]
            cls_queries = random.sample(all_cls_queries, self.episode_hparams.num_queries_per_class)

        else:
            cls_samples = random.sample(
                all_cls_supports,
                self.episode_hparams.num_supports_per_class + self.episode_hparams.num_queries_per_class,
            )
            cls_supports = cls_samples[: self.episode_hparams.num_supports_per_class]
            cls_queries = cls_samples[self.episode_hparams.num_supports_per_class :]

        return {"supports": cls_supports, "queries": cls_queries}


class IterableEpisodicDataset(torch.utils.data.IterableDataset, EpisodicDataset):
    def __init__(
        self,
        num_episodes: int,
        samples: List,
        stage_labels: List,
        class_to_label_dict: Dict,
        episode_hparams: EpisodeHParams,
        separated_query_support: bool,
    ):
        """ """
        super().__init__(
            num_episodes=num_episodes,
            samples=samples,
            stage_labels=stage_labels,
            class_to_label_dict=class_to_label_dict,
            episode_hparams=episode_hparams,
            separated_query_support=separated_query_support,
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            per_worker = self.num_episodes
            worker_id = 0
        else:  # in a worker process
            worker_id = worker_info.id
            # split workload
            per_worker = math.ceil(int(self.num_episodes / float(worker_info.num_workers)))

        random.seed(worker_id)

        return iter(self.sample_episode() for _ in range(per_worker))

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError


class MapEpisodicDataset(Dataset, EpisodicDataset):
    def __init__(
        self,
        num_episodes: int,
        samples: List,
        class_to_label_dict: Dict,
        stage_labels: List,
        episode_hparams: EpisodeHParams,
        separated_query_support: bool,
    ):
        """ """
        super().__init__(
            num_episodes=num_episodes,
            samples=samples,
            stage_labels=stage_labels,
            class_to_label_dict=class_to_label_dict,
            episode_hparams=episode_hparams,
            separated_query_support=separated_query_support,
        )

        self.episodes = [self.sample_episode() for _ in range(self.num_episodes)]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


class EpisodicDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, episode_hparams, **kwargs):
        collate_fn = partial(EpisodeBatch.from_episode_list, episode_hparams=episode_hparams)
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    _: EpisodicDataset = hydra.utils.instantiate(cfg.nn.data.datasets.train, split="train", _recursive_=False)


if __name__ == "__main__":
    main()
