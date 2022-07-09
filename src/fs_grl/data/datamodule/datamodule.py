import json
import logging
from abc import ABC
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import networkx as nx
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.data import Data

from fs_grl.data.datamodule.metadata import MetaData
from fs_grl.data.episode.episode import EpisodeHParams
from fs_grl.data.io_utils import (
    data_list_to_graph_list,
    get_classes_to_label_dict,
    graph_list_to_data_list,
    load_csv_data,
    load_graph_list,
    load_pickle_data,
    map_classes_to_labels,
)
from fs_grl.data.utils import flatten, get_label_to_samples_map, random_split_bucketed

pylogger = logging.getLogger(__name__)


class GraphFewShotDataModule(pl.LightningDataModule, ABC):
    def __init__(
        self,
        dataset_name,
        data_dir,
        feature_params: Dict,
        classes_split_path: Optional[str],
        train_ratio,
        test_episode_hparams: EpisodeHParams,
        num_test_episodes,
        batch_size: DictConfig,
        num_workers: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        **kwargs,
    ):
        """
        Abstract datamodule for few-shot graph classification. Only accounts for episodes at testing time,
        as it is the case for example for transfer learning.

        :param dataset_name:
        :param feature_params: what feature params to consider
        :param data_dir: path to the folder containing the dataset

        :param classes_split_path: path containing the split between base and novel classes

        :param train_ratio: percentage of base samples that go into training

        :param test_episode_hparams: number N of classes per episode, number K of supports per class and
                                number Q of queries per class
        :param num_test_episodes: how many episodes for testing


        :param batch_size:
        :param num_workers:
        :param gpus:

        :param kwargs:
        """
        super().__init__()

        self.classes_split_path = classes_split_path

        self.test_episode_hparams = test_episode_hparams
        self.num_test_episodes = num_test_episodes

        self.train_ratio = train_ratio

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory: bool = gpus is not None and str(gpus) != "0"

        assert batch_size.test == 1

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        data = self.load_data(data_dir, dataset_name, feature_params)
        self.data_list, self.graph_list = data["data_list"], data["graph_list"]
        self.classes_split, self.class_to_label_dict = data["classes_split"], data["class_to_label_dict"]

        self.label_to_class_dict: Dict[int, str] = {v: k for k, v in self.class_to_label_dict.items()}

        self.base_labels, self.novel_labels = self.labels_split["base"], self.labels_split["novel"]
        self.val_labels = self.labels_split["val"] if "val" in self.labels_split.keys() else self.base_labels

        self.data_list_by_label: Dict[int, List[Data]] = get_label_to_samples_map(self.data_list)
        self.graph_list_by_label: Dict[int, List[nx.Graph]] = get_label_to_samples_map(self.graph_list)

        self.data_list_by_base_label = {
            label: data_list for label, data_list in self.data_list_by_label.items() if label in self.base_labels
        }
        self.data_list_by_novel_label = {
            label: data_list for label, data_list in self.data_list_by_label.items() if label in self.novel_labels
        }
        self.graph_list_by_base_label = {
            label: graph_list for label, graph_list in self.graph_list_by_label.items() if label in self.base_labels
        }

        self.print_dataset_info()

    def load_data(self, data_dir, dataset_name, feature_params) -> Dict:
        """
        Load data from disk.
        """

        # TODO: convert pickle to txt and use only the latter
        if dataset_name in {"TRIANGLES", "ENZYMES", "Letter_high", "Reddit", "DUMMY"}:
            # handle txt data
            classes_split = self.get_classes_split()

            graph_list: List[nx.Graph] = load_graph_list(data_dir, dataset_name)

            class_to_label_dict = get_classes_to_label_dict(graph_list)
            map_classes_to_labels(graph_list, class_to_label_dict)

            data_list = graph_list_to_data_list(
                graph_list=graph_list,
                feature_params=feature_params,
            )

        elif dataset_name in {"COIL-DEL", "R52"}:
            # handle pickle data
            data_list, classes_split = load_pickle_data(
                data_dir=data_dir,
                dataset_name=dataset_name,
                feature_params=feature_params,
            )
            graph_list: List[nx.Graph] = data_list_to_graph_list(data_list)

            class_to_label_dict = {str(cls): cls for classes in classes_split.values() for cls in classes}

        elif dataset_name in {"Tox21"}:  # , "SIDER"
            classes_split = self.get_classes_split()

            data_list = load_csv_data(data_dir=data_dir, dataset_name=dataset_name, feature_params=feature_params)

            graph_list: List[nx.Graph] = data_list_to_graph_list(data_list)

            classes = [cls for classes in classes_split.values() for cls in classes]
            class_to_label_dict = {cls: ind for ind, cls in enumerate(classes)}

        else:
            raise NotImplementedError

        return {
            "graph_list": graph_list,
            "data_list": data_list,
            "classes_split": classes_split,
            "class_to_label_dict": class_to_label_dict,
        }

    @property
    def labels_split(self):
        return {
            split: sorted([self.class_to_label_dict[str(cls)] for cls in classes])
            for split, classes in self.classes_split.items()
        }

    @property
    def metadata(self) -> MetaData:
        """
        Data information to be fed to the Lightning Module as parameter.

        :return everything the model should know about the data, wrapped in a MetaData object.
        """
        # Since MetaData depends on the training data, we need to ensure the setup method has been called.
        if self.train_dataset is None:
            self.setup(stage="fit")

        metadata = MetaData(
            class_to_label_dict=self.class_to_label_dict,
            feature_dim=self.feature_dim,
            num_classes_per_episode=self.test_episode_hparams.num_classes_per_episode,
            classes_split=self.classes_split,
        )

        return metadata

    @property
    def feature_dim(self) -> int:
        ref_data = self.data_list[0]
        return ref_data.x.shape[-1]

    def get_classes_split(self) -> Dict:
        """
        Returns the classes split from file if present, else creates and saves a new one.
        """
        if self.classes_split_path is not None:
            classes_split = json.loads(Path(self.classes_split_path).read_text(encoding="utf-8"))
            return classes_split

        pylogger.info("No classes split provided, creating new split.")
        raise NotImplementedError

    def split_base_novel_samples(self) -> Dict[str, List[Data]]:
        """
        Split the samples in base and novel ones according to the labels
        """
        base_samples: List[List[Data]] = [samples for samples in self.data_list_by_base_label.values()]
        base_samples: List[Data] = flatten(base_samples)

        novel_samples: List[List[Data]] = [samples for samples in self.data_list_by_novel_label.values()]
        novel_samples: List[Data] = flatten(novel_samples)

        if "val" in self.labels_split.keys():
            val_samples: List[List[Data]] = [
                samples for key, samples in self.data_list_by_label.items() if key in self.val_labels
            ]
            val_samples = flatten(val_samples)
        else:
            base_samples, val_samples = random_split_bucketed(base_samples, self.train_ratio)

        return {"base": base_samples, "val": val_samples, "novel": novel_samples}

    def print_dataset_info(self):
        pylogger.info(f"Read {len(self.data_list)} graphs.")
        pylogger.info(f"With label distribution: {Counter(sample.y.item() for sample in self.data_list)}")
        pylogger.info(f"Class to dict mapping: {self.class_to_label_dict}")
        pylogger.info(
            f"Base labels:\n{self.base_labels}\nValidation labels:"
            f"\n{self.val_labels}\nnovel labels:\n{self.novel_labels}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.num_workers=}, " f"{self.batch_size=})"
