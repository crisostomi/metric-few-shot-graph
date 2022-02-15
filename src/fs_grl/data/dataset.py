import math
import operator
import pprint
import random
from abc import ABC
from itertools import groupby
from typing import Dict, List, Tuple, Union

import hydra
import omegaconf
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch_geometric.data import Batch, Data

from nn_core.common import PROJECT_ROOT

from fs_grl.data.episode import Episode, EpisodeBatch
from fs_grl.data.utils import AnnotatedSample


def get_cls_to_samples_map(annotated_samples: List) -> Dict[int, List[Data]]:
    """
    Given a list of annotated_samples, return a map { class: list of samples of that class}
    """
    res = {}
    for sample in annotated_samples:
        res.setdefault(sample.y.item(), []).append(sample)
    return res


class TransferSourceDataset(Dataset):
    def __init__(self, samples, class_to_label_dict, stage_labels):
        self.samples = samples
        self.class_to_label_dict = class_to_label_dict
        self.stage_labels = stage_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class EpisodicDataset(ABC):
    def __init__(
        self,
        n_episodes: int,
        samples: Union[List, Dict],
        stage_labels: set,
        class_to_label_dict: Dict,
        num_classes_per_episode,
        num_supports_per_class,
        num_queries_per_class,
        separated_query_support,
    ):
        """

        :param n_episodes:
        :param samples:
        :param queries:
        """
        super().__init__()
        self.n_episodes = n_episodes
        self.separated_query_support = separated_query_support

        self.num_classes_per_episode = num_classes_per_episode
        self.num_supports_per_class = num_supports_per_class
        self.num_queries_per_class = num_queries_per_class

        if self.separated_query_support:
            self.supports, self.queries = samples["supports"], samples["queries"]
        else:
            self.supports, self.queries = samples, None

        self.stage_labels = stage_labels
        self.class_to_label_dict = class_to_label_dict
        self.label_set = set([label for label in self.class_to_label_dict.values() if label in self.stage_labels])

        self.cls_to_supports: Dict[int : List[Data]] = get_cls_to_samples_map(self.supports)
        self.cls_to_queries: Dict[int : List[Data]] = (
            get_cls_to_samples_map(self.queries) if self.separated_query_support else None
        )

    def sample_episode(self):
        f"""
        Creates an episode by first sampling {self.num_classes_per_episode} classes
        and then sampling K supports and Q queries for each class
        :return:
        """
        labels = sorted(random.sample(self.label_set, self.num_classes_per_episode))

        supports = []
        queries = []

        for _, label in enumerate(labels):
            label_supports, label_queries = self.sample_label_queries_supports(label)

            supports += label_supports
            queries += label_queries

        random.shuffle(supports)
        random.shuffle(queries)

        episode_hparams = {
            "num_supports_per_class": self.num_supports_per_class,
            "num_queries_per_class": self.num_queries_per_class,
            "num_classes_per_episode": self.num_classes_per_episode,
        }

        return Episode(supports, queries, labels, episode_hparams=episode_hparams)

    def sample_label_queries_supports(self, label):
        all_cls_supports: List[Data] = self.cls_to_supports[label]

        if self.separated_query_support:
            cls_supports = random.sample(all_cls_supports, self.num_supports_per_class)

            all_cls_queries: List[Data] = self.cls_to_queries[label]
            cls_queries = random.sample(all_cls_queries, self.num_queries_per_class)

        else:
            cls_samples = random.sample(all_cls_supports, self.num_supports_per_class + self.num_queries_per_class)
            cls_supports = cls_samples[: self.num_supports_per_class]
            cls_queries = cls_samples[self.num_supports_per_class :]

        return cls_supports, cls_queries


class IterableEpisodicDataset(torch.utils.data.IterableDataset, EpisodicDataset):
    def __init__(
        self,
        n_episodes: int,
        samples: List,
        stage_labels: set,
        class_to_label_dict: Dict,
        num_classes_per_episode,
        num_supports_per_class,
        num_queries_per_class,
        separated_query_support,
    ):
        """

        :param n_episodes:
        :param samples:
        :param queries:
        """
        super().__init__(
            n_episodes=n_episodes,
            samples=samples,
            stage_labels=stage_labels,
            class_to_label_dict=class_to_label_dict,
            num_classes_per_episode=num_classes_per_episode,
            num_queries_per_class=num_queries_per_class,
            num_supports_per_class=num_supports_per_class,
            separated_query_support=separated_query_support,
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            per_worker = self.n_episodes
            worker_id = 0
        else:  # in a worker process
            worker_id = worker_info.id
            # split workload
            per_worker = math.ceil(int(self.n_episodes / float(worker_info.num_workers)))

        random.seed(worker_id)

        return iter(self.sample_episode() for _ in range(per_worker))


class MapEpisodicDataset(Dataset, EpisodicDataset):
    def __init__(
        self,
        n_episodes: int,
        samples: List,
        class_to_label_dict: Dict,
        stage_labels: set,
        num_classes_per_episode,
        num_supports_per_class,
        num_queries_per_class,
        separated_query_support,
    ):
        """
        :param n_episodes:
        :param samples:
        :param queries:
        """
        super().__init__(
            n_episodes=n_episodes,
            samples=samples,
            stage_labels=stage_labels,
            class_to_label_dict=class_to_label_dict,
            num_classes_per_episode=num_classes_per_episode,
            num_queries_per_class=num_queries_per_class,
            num_supports_per_class=num_supports_per_class,
            separated_query_support=separated_query_support,
        )

        self.episodes = [self.sample_episode() for _ in range(self.n_episodes)]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


class EpisodicDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, **kwargs):
        collate_fn = EpisodeBatch.from_episode_list
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    _: EpisodicDataset = hydra.utils.instantiate(cfg.nn.data.datasets.train, split="train", _recursive_=False)


if __name__ == "__main__":
    main()
