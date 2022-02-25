import logging
import math
import random
from abc import ABC
from functools import partial
from typing import Dict, List, Union

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from torch_geometric.data import Data

from nn_core.common import PROJECT_ROOT

from fs_grl.data.episode import Episode, EpisodeBatch, EpisodeHParams
from fs_grl.data.utils import get_label_to_samples_map

pylogger = logging.getLogger(__name__)


class VanillaGraphDataset(Dataset):
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

        labels = self.sample_labels()

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

    def sample_labels(self):
        sampled_labels = random.sample(self.stage_labels, self.episode_hparams.num_classes_per_episode)
        return sorted(sampled_labels)


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


class CurriculumIterableEpisodicDataset(IterableEpisodicDataset):
    def __init__(
        self,
        num_episodes: int,
        samples: List,
        stage_labels: List,
        class_to_label_dict: Dict,
        episode_hparams: EpisodeHParams,
        separated_query_support: bool,
        datamodule: pl.LightningDataModule,
        prototypes_path: str,
        max_difficult_step: int,
    ):
        super(CurriculumIterableEpisodicDataset, self).__init__(
            num_episodes=num_episodes,
            samples=samples,
            stage_labels=stage_labels,
            class_to_label_dict=class_to_label_dict,
            episode_hparams=episode_hparams,
            separated_query_support=separated_query_support,
        )

        self.datamodule = datamodule

        class_prototypes: Dict[str, torch.Tensor] = self.load_prototypes(prototypes_path)
        label_prototypes: Dict[int, torch.Tensor] = {
            self.class_to_label_dict[cls]: cls_prototype for cls, cls_prototype in class_prototypes.items()
        }
        label_similarity_matrix = self.compute_prototypes_similarities(label_prototypes)
        pylogger.info(f"Label similarity matrix: {label_similarity_matrix}")
        self.label_similarity_dict = self.matrix_to_similarity_dict(label_similarity_matrix)

        self.max_difficult_step = max_difficult_step

    def sample_labels(self):

        current_step = self.datamodule.trainer.global_step

        first_label = np.random.choice(self.stage_labels, size=1)
        sampled_labels = [first_label.tolist()[0]]

        for i in range(self.episode_hparams.num_classes_per_episode - 1):

            remaining_labels = set(self.stage_labels).difference(sampled_labels)
            remaining_labels_array = np.array(sorted(list(remaining_labels)))

            label_probabilities = self.get_label_probabilities(sampled_labels, remaining_labels, t=current_step)

            label = np.random.choice(remaining_labels_array, size=1, p=label_probabilities)

            sampled_labels.append(label[0])

        return sorted(sampled_labels)

    def compute_prototypes_similarities(self, label_prototypes):
        labels_prototypes = sorted([(k, v) for k, v in label_prototypes.items()], key=lambda t: t[0])
        prototypes = torch.stack([prototype for key, prototype in labels_prototypes])

        interleaved_repeated_prototypes = prototypes.repeat_interleave(len(self.stage_labels), dim=0)

        repeated_prototypes = prototypes.repeat((len(self.stage_labels), 1))

        similarities = torch.einsum(
            "qh,qh->q", (F.normalize(interleaved_repeated_prototypes), F.normalize(repeated_prototypes))
        ).reshape((len(self.stage_labels), len(self.stage_labels)))

        # similarities = similarities - torch.min(similarities, dim=-1)[0]
        # similarities = similarities / torch.max(similarities, dim=-1)[0]

        max_val = similarities.max(-1)[0].unsqueeze(dim=0).transpose(1, 0)
        min_val = -1 * torch.ones((len(self.stage_labels), 1))
        similarities = (similarities - min_val) / (max_val - min_val)
        return similarities

    def load_prototypes(self, prototypes_path):
        return torch.load(prototypes_path)

    def matrix_to_similarity_dict(self, label_similarity_matrix):
        label_similarity_dict = {}

        for ind, stage_label in enumerate(self.stage_labels):
            label_similarity_dict[stage_label] = {
                self.stage_labels[value_ind]: value
                for value_ind, value in enumerate(label_similarity_matrix[ind])
                if ind != value_ind
            }

        return label_similarity_dict

    def get_label_probabilities(self, sampled_labels, remaining_labels, t):

        label_weights = {}

        for label in remaining_labels:
            similarities_label_and_sampled = [
                sim.item() for lbl, sim in self.label_similarity_dict[label].items() if lbl in sampled_labels
            ]
            similarity_with_sampled = np.sum(np.exp(similarities_label_and_sampled))

            label_weight = self.weight_label(similarity_with_sampled, num_steps=t)

            label_weights[label] = label_weight

        norm_factor = sum(math.exp(j) for j in label_weights.values())
        label_probabilities = {k: math.exp(v) / norm_factor for k, v in label_weights.items()}
        label_probabilities_array = sorted([(k, v) for k, v in label_probabilities.items()], key=lambda t: t[0])
        label_probabilities_array = np.array([v for k, v in label_probabilities_array])

        return label_probabilities_array

    def weight_label(self, similarity_with_sampled, num_steps):
        t = min(num_steps / self.max_difficult_step, 1)

        return (1 - t) * (1 - similarity_with_sampled) + t * similarity_with_sampled


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
