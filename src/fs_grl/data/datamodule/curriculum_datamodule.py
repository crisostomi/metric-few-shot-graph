import logging
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig

from fs_grl.data.datamodule.episodic_datamodule import GraphEpisodicDataModule
from fs_grl.data.dataset.curriculum import CurriculumIterableEpisodicDataset

pylogger = logging.getLogger(__name__)


# TODO: get it back to work and refactor
class GraphCurriculumDataModule(GraphEpisodicDataModule):
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        feature_params: Dict,
        classes_split_path: Optional[str],
        episode_hparams: DictConfig,
        train_ratio: float,
        num_episodes_per_epoch: DictConfig,
        prototypes_path: str,
        max_difficult_step: int,
        batch_size: DictConfig,
        num_workers: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        **kwargs,
    ):

        super().__init__(
            dataset_name=dataset_name,
            data_dir=data_dir,
            feature_params=feature_params,
            episode_hparams=episode_hparams,
            classes_split_path=classes_split_path,
            num_episodes_per_epoch=num_episodes_per_epoch,
            train_ratio=train_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            gpus=gpus,
        )
        self.prototypes_path = prototypes_path
        self.max_difficult_step = max_difficult_step

    def get_train_dataset(self, train_dataset_params):
        dataset = CurriculumIterableEpisodicDataset(
            **train_dataset_params,
            datamodule=self,
        )
        return dataset
