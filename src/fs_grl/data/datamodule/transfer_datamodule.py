import logging
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, Union

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from fs_grl.data.datamodule.datamodule import GraphFewShotDataModule
from fs_grl.data.dataset.dataloader import EpisodicDataLoader
from fs_grl.data.dataset.episodic import MapEpisodicDataset
from fs_grl.data.dataset.vanilla import VanillaGraphDataset
from fs_grl.data.utils import random_split_sequence

pylogger = logging.getLogger(__name__)


class GraphTransferDataModule(GraphFewShotDataModule):
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
        Datamodule for Transfer Learning approaches. Training and validation are vanilla, i.e. non-episodic,
        and test is episodic.

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
        super().__init__(
            dataset_name=dataset_name,
            data_dir=data_dir,
            feature_params=feature_params,
            classes_split_path=classes_split_path,
            train_ratio=train_ratio,
            num_test_episodes=num_episodes_per_epoch.test,
            test_episode_hparams=episode_hparams.test,
            batch_size=batch_size,
            num_workers=num_workers,
            gpus=gpus,
        )

    def setup(self, stage: Optional[str] = None):

        if stage is None or stage == "fit":

            split_samples = self.split_base_novel_samples()
            base_samples, novel_samples = split_samples["base"], split_samples["novel"]

            base_samples, base_global_to_stage_labels = self.convert_to_stage_labels(base_samples, "base")
            pylogger.info(f"Base global to local labels: {base_global_to_stage_labels}")

            # TODO: currently taking a subset of the training dataset as validation set and ignoring the validation set
            # for the datasets for which it is available
            base_train_samples, base_val_samples = self.split_train_val(base_samples)

            self.train_dataset = VanillaGraphDataset(
                samples=base_train_samples,
            )

            self.val_datasets = [
                VanillaGraphDataset(
                    samples=base_val_samples,
                )
            ]

            novel_samples, novel_global_to_stage_labels = self.convert_to_stage_labels(novel_samples, "novel")
            pylogger.info(f"Novel global to local labels: {novel_global_to_stage_labels}")

            stage_novel_labels = [ind for ind, label in enumerate(sorted(self.novel_labels))]

            self.test_datasets = [
                MapEpisodicDataset(
                    samples=novel_samples,
                    num_episodes=self.num_test_episodes,
                    stage_labels=stage_novel_labels,
                    episode_hparams=self.test_episode_hparams,
                )
            ]

    def convert_to_stage_labels(self, samples: List[Data], base_or_novel: str) -> Tuple[List[Data], Dict]:
        """
        Given a list of samples, reassign their labels to be ordered from 0 to num_labels -1
        e.g. [2, 5, 10] --> [0, 1, 2]

        :param samples:
        :param base_or_novel: whether labels are base or novel ones

        :return samples with local labels and mapping
        """
        assert base_or_novel == "base" or base_or_novel == "novel"

        stage_labels = self.labels_split[base_or_novel]

        global_to_local_labels = {label: ind for ind, label in enumerate(sorted(stage_labels))}

        for sample in samples:
            sample.y.apply_(lambda x: global_to_local_labels[x])

        return samples, global_to_local_labels

    def split_train_val(self, data_list: List[Data]) -> Tuple[List[Data], List[Data]]:
        f"""
        Splits samples into training and validation according to {self.train_ratio}
        :return
        """

        train_samples, val_samples = random_split_sequence(sequence=data_list, split_ratio=self.train_ratio)

        pylogger.info(f"Train label dist: {Counter(sample.y.item() for sample in train_samples)}")
        pylogger.info(f"Val label dist: {Counter(sample.y.item() for sample in val_samples)}")

        return train_samples, val_samples

    # meta-training training
    def train_dataloader(self) -> DataLoader:

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size.train,
            collate_fn=Batch.from_data_list,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    # meta-training validation
    def val_dataloader(self) -> Sequence[DataLoader]:
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
        return [
            EpisodicDataLoader(
                dataset=dataset,
                episode_hparams=self.test_episode_hparams,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                pin_memory=self.pin_memory,
            )
            for dataset in self.test_datasets
        ]

    def predict_dataloader(self):
        pass
