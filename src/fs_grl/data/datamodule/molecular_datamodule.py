import json
import logging
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import networkx as nx
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.data import Data

from fs_grl.data.datamodule.metadata import MetaData
from fs_grl.data.dataset.dataloader import MolecularEpisodicDataLoader
from fs_grl.data.dataset.molecular import IterableMolecularDataset, MapMolecularDataset
from fs_grl.data.io_utils import data_list_to_graph_list, load_csv_data
from fs_grl.data.utils import DotDict

pylogger = logging.getLogger(__name__)


class MolecularDataModule(pl.LightningDataModule, ABC):
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        feature_params: Dict,
        classes_split_path: Optional[str],
        episode_hparams: DictConfig,
        train_ratio: float,
        num_episodes_per_epoch: DictConfig,
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

        self.episode_hparams = DotDict(
            {key: instantiate(val, _recursive_=True) for key, val in episode_hparams.items()}
        )

        self.properties_split_path = classes_split_path

        self.test_episode_hparams = self.episode_hparams.test
        self.num_episodes_per_epoch = num_episodes_per_epoch

        self.num_test_episodes = self.num_episodes_per_epoch.test

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
        self.properties_split, self.property_to_id_dict = data["classes_split"], data["class_to_label_dict"]

        self.id_to_property: Dict[int, str] = {v: k for k, v in self.property_to_id_dict.items()}

        self.base_properties, self.val_properties, self.novel_properties = (
            self.properties_id_split["base"],
            self.properties_id_split["val"],
            self.properties_id_split["novel"],
        )

        self.samples_by_property = self.get_samples_by_property()

    def load_data(self, data_dir, dataset_name, feature_params) -> Dict:
        """
        Load data from disk.
        """

        if dataset_name in {"Tox21"}:  # , "SIDER"
            properties_split = self.get_properties_split()

            data_list = load_csv_data(data_dir=data_dir, dataset_name=dataset_name, feature_params=feature_params)

            graph_list: List[nx.Graph] = data_list_to_graph_list(data_list)

            classes = [cls for classes in properties_split.values() for cls in classes]
            class_to_label_dict = {cls: ind for ind, cls in enumerate(classes)}

        else:
            raise NotImplementedError

        return {
            "graph_list": graph_list,
            "data_list": data_list,
            "classes_split": properties_split,
            "class_to_label_dict": class_to_label_dict,
        }

    def get_samples_by_property(self) -> Dict[int, Dict[str, List[Data]]]:
        samples_by_property = {}

        for property_id in self.property_to_id_dict.values():
            samples_by_property[property_id] = {"positive": [], "negative": []}

            for sample in self.data_list:

                sample_property_value = sample.y[0][property_id].item()
                new_sample = Data(
                    x=sample.x,
                    edge_index=sample.edge_index,
                    y=torch.tensor(sample_property_value).long(),
                )

                if sample_property_value == 0:
                    samples_by_property[property_id]["negative"].append(new_sample)
                elif sample_property_value == 1:
                    samples_by_property[property_id]["positive"].append(new_sample)

        return samples_by_property

    @property
    def properties_id_split(self):
        return {
            split: sorted([self.property_to_id_dict[str(cls)] for cls in classes])
            for split, classes in self.properties_split.items()
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
            class_to_label_dict=self.property_to_id_dict,
            feature_dim=self.feature_dim,
            num_classes_per_episode=self.test_episode_hparams.num_classes_per_episode,
            classes_split=self.properties_split,
        )

        return metadata

    @property
    def feature_dim(self) -> int:
        ref_data = self.data_list[0]
        return ref_data.x.shape[-1]

    def get_properties_split(self) -> Dict:
        """
        Returns the classes split from file if present, else creates and saves a new one.
        """
        if self.properties_split_path is not None:
            classes_split = json.loads(Path(self.properties_split_path).read_text(encoding="utf-8"))
            return classes_split

        pylogger.info("No classes split provided, creating new split.")
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.num_workers=}, " f"{self.batch_size=})"

    def setup(self, stage: Optional[str] = None):

        if stage is None or stage == "fit":

            train_dataset_params = {
                "samples_by_property": self.samples_by_property,
                "num_episodes": self.num_episodes_per_epoch.train,
                "stage_properties": self.base_properties,
                "episode_hparams": self.episode_hparams.train,
            }

            self.train_dataset = self.get_train_dataset(train_dataset_params)

            self.val_datasets = [
                MapMolecularDataset(
                    samples_by_property=self.samples_by_property,
                    num_episodes=self.num_episodes_per_epoch.val,
                    stage_properties=self.val_properties,
                    episode_hparams=self.episode_hparams.val,
                )
            ]

            self.test_datasets = [
                MapMolecularDataset(
                    samples_by_property=self.samples_by_property,
                    num_episodes=self.num_test_episodes,
                    stage_properties=self.novel_properties,
                    episode_hparams=self.episode_hparams.test,
                )
            ]

    def get_train_dataset(self, train_dataset_params):
        train_dataset = IterableMolecularDataset(**train_dataset_params)
        return train_dataset

    def train_dataloader(self) -> MolecularEpisodicDataLoader:
        return MolecularEpisodicDataLoader(
            dataset=self.train_dataset,
            episode_hparams=self.episode_hparams.train,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return [
            MolecularEpisodicDataLoader(
                dataset=dataset,
                episode_hparams=self.episode_hparams.val,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                pin_memory=self.pin_memory,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[MolecularEpisodicDataLoader]:
        return [
            MolecularEpisodicDataLoader(
                dataset=dataset,
                episode_hparams=self.episode_hparams.test,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                pin_memory=self.pin_memory,
            )
            for dataset in self.test_datasets
        ]

    def predict_dataloader(self):
        pass
