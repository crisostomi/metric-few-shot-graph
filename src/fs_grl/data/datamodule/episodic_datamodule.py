import logging
from typing import Dict, List, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig

from fs_grl.data.dataloader import EpisodicDataLoader
from fs_grl.data.datamodule.datamodule import GraphFewShotDataModule
from fs_grl.data.dataset.episodic import IterableEpisodicDataset, MapEpisodicDataset
from fs_grl.data.utils import DotDict

pylogger = logging.getLogger(__name__)


class GraphEpisodicDataModule(GraphFewShotDataModule):
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
        Datamodule for approaches following the episodic framework already at training time.
        This is used both for Distance Metric Learning and Meta-Learning.

        :param dataset_name:
        :param data_dir: path to the folder containing the dataset
        :param feature_params: what feature params to consider

        :param classes_split_path: path containing the split between base and novel classes


        :param episode_hparams: number N of classes per episode, number K of supports per class and
                                number Q of queries per class for train, val and test

        :param train_ratio: percentage of base samples that go into training

        :param num_episodes_per_epoch: how many episodes for epoch for train, val and test

        :param batch_size:
        :param num_workers:
        :param gpus:

        :param kwargs:
        """
        self.episode_hparams = DotDict(
            {key: instantiate(val, _recursive_=True) for key, val in episode_hparams.items()}
        )

        super().__init__(
            dataset_name=dataset_name,
            feature_params=feature_params,
            data_dir=data_dir,
            classes_split_path=classes_split_path,
            test_episode_hparams=self.episode_hparams.test,
            num_test_episodes=num_episodes_per_epoch.test,
            train_ratio=train_ratio,
            num_workers=num_workers,
            batch_size=batch_size,
            gpus=gpus,
        )

        self.num_episodes_per_epoch = num_episodes_per_epoch

    def setup(self, stage: Optional[str] = None):

        if stage is None or stage == "fit":

            split_samples = self.split_base_novel_samples()
            base_samples, val_samples = split_samples["base"], split_samples["val"]

            train_dataset_params = {
                "samples": base_samples,
                "num_episodes": self.num_episodes_per_epoch.train,
                "stage_labels": self.base_labels,
                "episode_hparams": self.episode_hparams.train,
            }

            self.train_dataset = self.get_train_dataset(train_dataset_params)

            self.val_datasets = [
                MapEpisodicDataset(
                    samples=val_samples,
                    num_episodes=self.num_episodes_per_epoch.val,
                    stage_labels=self.val_labels,
                    episode_hparams=self.episode_hparams.val,
                )
            ]

            novel_samples = split_samples["novel"]
            self.test_datasets = [
                MapEpisodicDataset(
                    samples=novel_samples,
                    num_episodes=self.num_test_episodes,
                    stage_labels=self.novel_labels,
                    episode_hparams=self.episode_hparams.test,
                )
            ]

    def get_train_dataset(self, train_dataset_params):
        train_dataset = IterableEpisodicDataset(**train_dataset_params)
        return train_dataset

    def train_dataloader(self) -> EpisodicDataLoader:
        return EpisodicDataLoader(
            dataset=self.train_dataset,
            episode_hparams=self.episode_hparams.train,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return [
            EpisodicDataLoader(
                dataset=dataset,
                episode_hparams=self.episode_hparams.val,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                pin_memory=self.pin_memory,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[EpisodicDataLoader]:
        return [
            EpisodicDataLoader(
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
