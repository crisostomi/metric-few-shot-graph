import json
import logging
import math
import operator
import random
from collections import Counter
from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from nn_core.common import PROJECT_ROOT

from fs_grl.data.dataset import EpisodicDataLoader, IterableEpisodicDataset, MapEpisodicDataset, TransferSourceDataset
from fs_grl.data.episode import EpisodeBatch, EpisodeHParams
from fs_grl.data.io_utils import load_data, load_query_support_idxs

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self, class_to_label_dict, feature_dim, episode_hparams: EpisodeHParams, classes_split: Dict):
        """The data information the Lightning Module will be provided with.
        This is a "bridge" between the Lightning DataModule and the Lightning Module.
        There is no constraint on the class name nor in the stored information, as long as it exposes the
        `save` and `load` methods.
        The Lightning Module will receive an instance of MetaData when instantiated,
        both in the train loop or when restored from a checkpoint.
        This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
        DataModule/Trainer independent (useful in prediction scenarios).
        MetaData should contain all the information needed at test time, derived from its train dataset.
        Examples are the class names in a classification task or the vocabulary in NLP tasks.
        MetaData exposes `save` and `load`. Those are two user-defined methods that specify
        how to serialize and de-serialize the information contained in its attributes.
        This is needed for the checkpointing restore to work properly.
        Args:
            class_vocab: association between class names and their indices
        """
        self.classes_to_label_dict = class_to_label_dict
        self.feature_dim = feature_dim
        self.num_classes = len(class_to_label_dict)
        self.episode_hparams = episode_hparams
        self.classes_split = classes_split

    def save(self, dst_path: Path) -> None:
        """Serialize the MetaData attributes into the zipped checkpoint in dst_path.
        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        data = {
            "classes_to_label_dict": self.classes_to_label_dict,
            "feature_dim": self.feature_dim,
            "episode_hparams": self.episode_hparams,
            "classes_split": self.classes_split,
        }

        (dst_path / "data.json").write_text(json.dumps(data, indent=4, default=lambda x: x.__dict__))

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.
        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint
        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        data = json.loads((src_path / "data.json").read_text(encoding="utf-8"))

        return MetaData(
            class_to_label_dict=data["classes_to_label_dict"],
            feature_dim=data["feature_dim"],
            episode_hparams=EpisodeHParams(**data["episode_hparams"]),
            classes_split=data["classes_split"],
        )


class GraphFewShotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name,
        data_dir,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        num_supports_per_class,
        classes_split_path: Optional[str] = None,
        query_support_path: str = None,
        query_support_ratio: float = 0.8,
        num_episodes: int = 2000,
        num_queries_per_class: int = 2,
        num_classes_per_episode: int = 2,
        separated_query_support: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.num_queries_per_class = num_queries_per_class
        self.pin_memory: bool = gpus is not None and str(gpus) != "0"
        self.num_supports_per_class = num_supports_per_class

        self.episode_hparams = EpisodeHParams(num_classes_per_episode, num_supports_per_class, num_queries_per_class)

        self.query_support_path = query_support_path
        self.support_query_ratio = query_support_ratio
        self.num_classes_per_episode = num_classes_per_episode
        self.separated_query_support = separated_query_support

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.classes_split_path = classes_split_path
        self.classes_split = self.get_classes_split()
        self.base_classes, self.novel_classes = self.classes_split["base"], self.classes_split["novel"]

        self.data_list, self.class_to_label_dict = load_data(self.data_dir, self.dataset_name, attr_to_consider="both")

        self.base_labels = set([self.class_to_label_dict[stage_cls] for stage_cls in self.base_classes])
        self.novel_labels = set([self.class_to_label_dict[stage_cls] for stage_cls in self.novel_classes])

        self.labels_split = {"base": self.base_labels, "novel": self.novel_labels}

        self.feature_dim = self.data_list[0].x.shape[-1]

        self.data_list_by_label = {
            key.item(): list(value) for key, value in groupby(self.data_list, key=operator.attrgetter("y"))
        }

        self.num_classes = len(self.class_to_label_dict)

    @property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.
        Examples are vocabularies, number of classes...
        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """
        # Since MetaData depends on the training data, we need to ensure the setup method has been called.
        if self.train_dataset is None:
            self.setup(stage="fit")

        metadata = MetaData(
            class_to_label_dict=self.class_to_label_dict,
            feature_dim=self.feature_dim,
            episode_hparams=self.episode_hparams,
            classes_split={"base": self.base_classes, "novel": self.novel_classes},
        )

        return metadata

    def get_classes_split(self):
        if self.classes_split_path is not None:
            classes_split = json.loads(Path(self.classes_split_path).read_text(encoding="utf-8"))
            return classes_split

        pylogger.info("No classes split provided, creating new split.")
        # TODO: implement
        pass

    def split_query_support(self, data_list: List[Data]):
        if self.query_support_path is not None:
            query_idxs, support_idxs = load_query_support_idxs(self.query_support_path)
            return query_idxs, support_idxs

        pylogger.info("No query support split provided, executing the split.")

        idxs = np.arange(len(data_list))
        random.shuffle(idxs)

        support_upperbound = math.ceil(self.support_query_ratio * len(data_list))
        support_idxs = idxs[:support_upperbound]
        query_idxs = idxs[support_upperbound:]

        return query_idxs, support_idxs

    def split_base_novel_samples(self) -> Tuple[List[Data], List[Data]]:
        base_samples: List[Data] = [
            sample for key, samples in self.data_list_by_label.items() if key in self.base_labels for sample in samples
        ]

        novel_samples = [
            sample for key, samples in self.data_list_by_label.items() if key in self.novel_labels for sample in samples
        ]
        return base_samples, novel_samples

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, " f"{self.batch_size=})"


class GraphMetaDataModule(GraphFewShotDataModule):
    def __init__(
        self,
        dataset_name,
        data_dir,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        num_supports_per_class,
        classes_split_path: Optional[str],
        query_support_path,
        query_support_ratio,
        num_episodes,
        num_queries_per_class,
        num_classes_per_episode,
        separated_query_support,
        **kwargs,
    ):

        super().__init__(
            dataset_name,
            data_dir,
            datasets,
            num_workers,
            batch_size,
            gpus,
            num_supports_per_class,
            classes_split_path,
            query_support_path,
            query_support_ratio,
            num_episodes,
            num_queries_per_class,
            num_classes_per_episode,
            separated_query_support,
            **kwargs,
        )

    def setup(self, stage: Optional[str] = None):

        if stage is None or stage == "fit":

            base_samples, novel_samples = self.split_base_novel_samples()

            if self.separated_query_support:
                base_query_idxs, base_support_idxs = self.split_query_support(base_samples)

                base_supports = [base_samples[idx] for idx in base_support_idxs]
                base_queries = [base_samples[idx] for idx in base_query_idxs]

                base_samples = {"supports": base_supports, "queries": base_queries}

            self.train_dataset = IterableEpisodicDataset(
                samples=base_samples,
                n_episodes=self.num_episodes,
                class_to_label_dict=self.class_to_label_dict,
                stage_labels=self.base_labels,
                num_supports_per_class=self.num_supports_per_class,
                num_queries_per_class=self.num_queries_per_class,
                num_classes_per_episode=self.num_classes_per_episode,
                separated_query_support=self.separated_query_support,
            )

            self.val_datasets = [
                MapEpisodicDataset(
                    samples=novel_samples,
                    n_episodes=self.num_episodes,
                    num_supports_per_class=self.num_supports_per_class,
                    stage_labels=self.novel_labels,
                    class_to_label_dict=self.class_to_label_dict,
                    num_queries_per_class=self.num_queries_per_class,
                    num_classes_per_episode=self.num_classes_per_episode,
                    separated_query_support=False,
                )
            ]

    def train_dataloader(self) -> DataLoader:
        collate_fn = partial(EpisodeBatch.from_episode_list, episode_hparams=self.episode_hparams)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size.train,
            collate_fn=collate_fn,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Sequence[EpisodicDataLoader]:
        collate_fn = partial(EpisodeBatch.from_episode_list, episode_hparams=self.episode_hparams)
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
            )
            for dataset in self.val_datasets
        ]


class GraphTransferDataModule(GraphFewShotDataModule):
    def __init__(
        self,
        dataset_name,
        data_dir,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        num_supports_per_class,
        train_val_split_ratio=0.8,
        **kwargs,
    ):
        super().__init__(
            dataset_name, data_dir, datasets, num_workers, batch_size, gpus, num_supports_per_class, **kwargs
        )
        self.train_val_split_ratio = train_val_split_ratio

    def setup(self, stage: Optional[str] = None):

        if stage is None or stage == "fit":

            base_samples, novel_samples = self.split_base_novel_samples()

            self.convert_to_local_labels(base_samples, "base")

            base_train_samples, base_val_samples = self.split_train_val(base_samples)

            self.train_dataset = TransferSourceDataset(
                samples=base_train_samples,
                class_to_label_dict=self.class_to_label_dict,
                stage_labels=self.base_labels,
            )

            self.val_datasets = [
                TransferSourceDataset(
                    samples=base_val_samples,
                    stage_labels=self.base_labels,
                    class_to_label_dict=self.class_to_label_dict,
                )
            ]
            # self.convert_to_local_labels(novel_samples, "novel")

    def convert_to_local_labels(self, samples, base_or_novel):

        stage_labels = self.labels_split[base_or_novel]

        self.global_to_local_labels = {label: ind for ind, label in enumerate(sorted(stage_labels))}

        for sample in samples:
            sample.y.apply_(lambda x: self.global_to_local_labels[x])

    def split_train_val(self, data_list):
        idxs = np.arange(len(data_list))
        random.shuffle(idxs)

        train_upperbound = math.ceil(self.train_val_split_ratio * len(data_list))
        train_idxs = idxs[:train_upperbound]
        val_idxs = idxs[train_upperbound:]

        train_samples = [data_list[idx] for idx in train_idxs]
        val_samples = [data_list[idx] for idx in val_idxs]

        print(f"Train label dist: {Counter(sample.y.item() for sample in train_samples)}")
        print(f"Val label dist: {Counter(sample.y.item() for sample in val_samples)}")
        return train_samples, val_samples

    # meta-training training
    def train_dataloader(self) -> DataLoader:

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size.train,
            collate_fn=Batch.from_data_list,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
        )

    # meta-training validation
    def val_dataloader(self) -> Sequence[EpisodicDataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                pin_memory=self.pin_memory,
                collate_fn=Batch.from_data_list,
            )
            for dataset in self.val_datasets
        ]

    # meta-testing
    def test_dataloader(self) -> Sequence[EpisodicDataLoader]:
        collate_fn = partial(EpisodeBatch.from_episode_list, episode_hparams=self.episode_hparams)
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=1,
                num_workers=self.num_workers.test,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
            )
            for dataset in self.test_datasets
        ]


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    datamodule.setup()
    for x in datamodule.train_dataloader():
        print(x)


if __name__ == "__main__":
    main()
