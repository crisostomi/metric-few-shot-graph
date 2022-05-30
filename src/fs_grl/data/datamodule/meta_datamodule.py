import logging
from typing import Dict, List, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig

from fs_grl.data.datamodule.datamodule import GraphFewShotDataModule
from fs_grl.data.dataset import (
    CurriculumIterableEpisodicDataset,
    EpisodicDataLoader,
    IterableEpisodicDataset,
    MapEpisodicDataset,
)
from fs_grl.data.episode import EpisodeHParams

pylogger = logging.getLogger(__name__)


class GraphMetaDataModule(GraphFewShotDataModule):
    def __init__(
        self,
        dataset_name,
        feature_params: Dict,
        data_dir,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        classes_split_path: Optional[str],
        query_support_split_path,
        train_episode_hparams: EpisodeHParams,
        val_episode_hparams: EpisodeHParams,
        test_episode_hparams: EpisodeHParams,
        support_ratio,
        train_ratio,
        num_train_episodes,
        num_val_episodes,
        num_test_episodes,
        separated_query_support,
        curriculum_learning: bool,
        prototypes_path: str = "",
        max_difficult_step: int = 0,
        **kwargs,
    ):

        super().__init__(
            dataset_name=dataset_name,
            feature_params=feature_params,
            data_dir=data_dir,
            classes_split_path=classes_split_path,
            query_support_split_path=query_support_split_path,
            test_episode_hparams=test_episode_hparams,
            num_test_episodes=num_test_episodes,
            separated_query_support=separated_query_support,
            support_ratio=support_ratio,
            train_ratio=train_ratio,
            num_workers=num_workers,
            batch_size=batch_size,
            gpus=gpus,
        )

        self.train_episode_hparams = instantiate(train_episode_hparams)
        self.val_episode_hparams = instantiate(val_episode_hparams)

        self.num_train_episodes = num_train_episodes
        self.num_val_episodes = num_val_episodes

        self.curriculum_learning = curriculum_learning
        self.prototypes_path = prototypes_path
        self.max_difficult_step = max_difficult_step

    def setup(self, stage: Optional[str] = None):

        if stage is None or stage == "fit":

            split_samples = self.split_base_novel_samples()
            base_samples, val_samples = split_samples["base"], split_samples["val"]

            if self.separated_query_support:
                base_samples = self.split_query_support(base_samples)

            train_dataset_params = {
                "samples": base_samples,
                "num_episodes": self.num_train_episodes,
                "class_to_label_dict": self.class_to_label_dict,
                "stage_labels": self.base_labels,
                "episode_hparams": self.train_episode_hparams,
                "separated_query_support": self.separated_query_support,
            }

            if self.curriculum_learning:
                self.train_dataset = CurriculumIterableEpisodicDataset(
                    **train_dataset_params,
                    datamodule=self,
                    prototypes_path=self.prototypes_path,
                    max_difficult_step=self.max_difficult_step,
                )
            else:
                self.train_dataset = IterableEpisodicDataset(**train_dataset_params)

            self.val_datasets = [
                MapEpisodicDataset(
                    samples=val_samples,
                    num_episodes=self.num_val_episodes,
                    stage_labels=self.val_labels,
                    class_to_label_dict=self.class_to_label_dict,
                    episode_hparams=self.val_episode_hparams,
                    separated_query_support=False,
                )
            ]

            novel_samples = split_samples["novel"]
            self.test_datasets = [
                MapEpisodicDataset(
                    samples=novel_samples,
                    num_episodes=self.num_test_episodes,
                    stage_labels=self.novel_labels,
                    class_to_label_dict=self.class_to_label_dict,
                    episode_hparams=self.test_episode_hparams,
                    separated_query_support=False,
                )
            ]

    def train_dataloader(self) -> EpisodicDataLoader:
        return EpisodicDataLoader(
            dataset=self.train_dataset,
            episode_hparams=self.train_episode_hparams,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return [
            EpisodicDataLoader(
                dataset=dataset,
                episode_hparams=self.val_episode_hparams,
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
