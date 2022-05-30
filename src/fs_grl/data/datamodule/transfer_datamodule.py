import logging
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, Union

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from fs_grl.data.datamodule.datamodule import GraphFewShotDataModule
from fs_grl.data.dataset import EpisodicDataLoader, MapEpisodicDataset, VanillaGraphDataset
from fs_grl.data.episode import EpisodeHParams
from fs_grl.data.utils import random_split_sequence

pylogger = logging.getLogger(__name__)


class GraphTransferDataModule(GraphFewShotDataModule):
    def __init__(
        self,
        dataset_name,
        feature_params: Dict,
        data_dir,
        classes_split_path: Optional[str],
        query_support_split_path,
        separated_query_support: bool,
        support_ratio,
        test_episode_hparams: EpisodeHParams,
        num_test_episodes,
        train_ratio,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        **kwargs,
    ):
        super().__init__(
            dataset_name=dataset_name,
            feature_params=feature_params,
            data_dir=data_dir,
            classes_split_path=classes_split_path,
            query_support_split_path=query_support_split_path,
            separated_query_support=separated_query_support,
            support_ratio=support_ratio,
            train_ratio=train_ratio,
            num_test_episodes=num_test_episodes,
            test_episode_hparams=test_episode_hparams,
            num_workers=num_workers,
            batch_size=batch_size,
            gpus=gpus,
            add_artificial_nodes=False,
            artificial_node_features=None,
        )

    def setup(self, stage: Optional[str] = None):

        if stage is None or stage == "fit":

            split_samples = self.split_base_novel_samples()
            base_samples, novel_samples = split_samples["base"], split_samples["novel"]

            base_samples, base_global_to_local_labels = self.convert_to_local_labels(base_samples, "base")
            pylogger.info(f"Base global to local labels: {base_global_to_local_labels}")

            base_train_samples, base_val_samples = self.split_train_val(base_samples)

            self.train_dataset = VanillaGraphDataset(
                samples=base_train_samples,
            )

            self.val_datasets = [
                VanillaGraphDataset(
                    samples=base_val_samples,
                )
            ]

            novel_samples, novel_global_to_local_labels = self.convert_to_local_labels(novel_samples, "novel")
            pylogger.info(f"Novel global to local labels: {novel_global_to_local_labels}")

            local_novel_labels = [ind for ind, label in enumerate(sorted(self.novel_labels))]

            self.test_datasets = [
                MapEpisodicDataset(
                    samples=novel_samples,
                    num_episodes=self.num_test_episodes,
                    stage_labels=local_novel_labels,
                    class_to_label_dict=self.class_to_label_dict,
                    episode_hparams=self.test_episode_hparams,
                    separated_query_support=False,
                )
            ]

    def convert_to_local_labels(self, samples: List[Data], base_or_novel: str) -> Tuple[List[Data], Dict]:
        """
        Given a list of samples, reassign their labels to be ordered from 0 to num_labels -1
        e.g. [2, 5, 10] --> [0, 1, 2]
        :param samples:
        :param base_or_novel: whether labels are base or novel ones
        :return: samples with local labels and mapping
        """
        stage_labels = self.labels_split[base_or_novel]

        global_to_local_labels = {label: ind for ind, label in enumerate(sorted(stage_labels))}

        for sample in samples:
            sample.y.apply_(lambda x: global_to_local_labels[x])

        return samples, global_to_local_labels

    def split_train_val(self, data_list: List[Data]) -> Tuple[List[Data], List[Data]]:
        f"""
        Splits samples into training and validation according to {self.train_ratio}
        :return:
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
                batch_size=1,
                num_workers=self.num_workers.test,
                pin_memory=self.pin_memory,
            )
            for dataset in self.test_datasets
        ]

    def predict_dataloader(self):
        pass
