import itertools
import logging
from typing import Any, Mapping, Optional

import torch
from hydra.utils import instantiate
from torch import nn
from torchmetrics import Accuracy, FBetaScore

from nn_core.model_logging import NNLogger

from fs_grl.data.datamodule import MetaData
from fs_grl.data.episode import EpisodeBatch
from fs_grl.pl_modules.pl_module import MyLightningModule
from fs_grl.utils import compute_global_prototypes

pylogger = logging.getLogger(__name__)


class DistanceMetricLearning(MyLightningModule):
    logger: NNLogger

    def __init__(self, train_data_list_by_label, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata

        # classes should be sorted, might add an assert later
        self.classes = list(metadata.classes_to_label_dict.keys())
        self.label_to_class_dict = {v: k for k, v in metadata.classes_to_label_dict.items()}

        self.model = instantiate(
            self.hparams.model,
            cfg=self.hparams.model,
            feature_dim=self.metadata.feature_dim,
            num_classes=self.metadata.num_classes,
            num_classes_per_episode=self.metadata.num_classes_per_episode,
            _recursive_=False,
        )

        reductions = ["micro", "weighted", "macro", "none"]
        metrics = (("F1", FBetaScore), ("acc", Accuracy))
        self.val_metrics = nn.ModuleDict(
            {
                f"val/{metric_name}/{reduction}": metric(num_classes=self.metadata.num_classes, average=reduction)
                for reduction, (metric_name, metric) in itertools.product(reductions, metrics)
            }
        )
        self.test_metrics = nn.ModuleDict(
            {
                f"test/{metric_name}/{reduction}": metric(num_classes=self.metadata.num_classes, average=reduction)
                for reduction, (metric_name, metric) in itertools.product(reductions, metrics)
            }
        )
        self.train_metrics = nn.ModuleDict({"train/acc/micro": Accuracy(num_classes=self.metadata.num_classes)})

        self.base_prototypes = {}
        self.train_data_list_by_label = train_data_list_by_label
        # metrics computed without mapping
        # self.test_metrics = nn.ModuleDict({"test/micro_acc": Accuracy(num_classes=metadata.num_classes_per_episode)})
        # self.val_metrics = nn.ModuleDict({"val/micro_acc": Accuracy(num_classes=metadata.num_classes_per_episode)})

    def forward(self, batch: EpisodeBatch) -> torch.Tensor:
        """
        :return similarities, tensor ~ (B*(N*Q)*N) containing for each episode the similarity
                between each of the N*Q queries and the N label prototypes
        """

        model_out = self.model(batch)

        return model_out

    def step(self, batch, split: str) -> Mapping[str, Any]:

        model_out = self(batch)

        loss = self.model.compute_loss(model_out, batch)

        self.log_dict({f"loss/{split}": loss}, on_epoch=True, on_step=True)

        return {"similarities": model_out["similarities"], "loss": loss}

    def training_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch, "train")
        similarities = step_out["similarities"]

        predictions = self.get_predictions(similarities, batch)

        for metric_name, metric in self.train_metrics.items():
            metric_res = metric(preds=predictions, target=batch.queries.y)
            self.log(name=metric_name, value=metric_res, on_step=True, on_epoch=True)

        return step_out

    def on_train_end(self) -> None:
        base_prototypes = compute_global_prototypes(self, self.train_data_list_by_label, self.label_to_class_dict)
        self.base_prototypes = base_prototypes

    def validation_step(self, batch: EpisodeBatch, batch_idx: int):
        step_out = self.step(batch, "val")

        # shape (B*(N*Q)*N)
        similarities = step_out["similarities"]

        predictions = self.get_predictions(similarities, batch)

        for metric_name, metric in self.val_metrics.items():
            metric(preds=predictions, target=batch.queries.y)

        self.log_metrics(split="val", on_step=True, on_epoch=True, cm_reset=False)

        return step_out

    def test_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch, "test")

        # shape ~(num_episodes * num_queries_per_class * num_classes_per_episode)
        similarities = step_out["similarities"]

        predictions = self.get_predictions(similarities, batch)

        for metric_name, metric in self.test_metrics.items():
            metric(preds=predictions, target=batch.queries.y)

        self.log_metrics(split="test", on_step=True, on_epoch=True, cm_reset=False)

        return step_out

    def get_predictions(self, similarities: torch.Tensor, batch: EpisodeBatch) -> torch.Tensor:
        """

        :param similarities: shape (B * N*Q * N)
        :param batch:
        :return:
        """
        num_classes_per_episode = batch.episode_hparams.num_classes_per_episode

        # shape (B*(N*Q), N) contains the similarity between the query
        # and the N label prototypes for each of the N*Q queries
        similarities_per_label = similarities.reshape((-1, num_classes_per_episode))

        # shape (B*(N*Q)) contains for each query the most similar label
        pred_labels = torch.argmax(similarities_per_label, dim=-1)

        pred_global_labels = self.map_pred_labels_to_global(
            pred_labels=pred_labels, batch_global_labels=batch.global_labels, num_episodes=batch.num_episodes
        )

        return pred_global_labels

    @staticmethod
    def map_pred_labels_to_global(pred_labels, batch_global_labels, num_episodes):
        """

        :param pred_labels: (B*N*Q)
        :param batch_global_labels: (B*N)
        :param num_episodes: number of episodes in the batch

        :return:
        """
        global_labels_per_episode = batch_global_labels.reshape(num_episodes, -1)
        pred_labels = pred_labels.reshape(num_episodes, -1)

        mapped_labels = []
        for episode_num in range(num_episodes):

            # shape (N)
            episode_global_labels = global_labels_per_episode[episode_num]
            # shape (N*Q)
            episode_pred_labels = pred_labels[episode_num]
            # shape (N*Q)
            episode_mapped_labels = episode_global_labels[episode_pred_labels]

            mapped_labels.append(episode_mapped_labels)

        # shape (B*N*Q)
        mapped_labels = torch.cat(mapped_labels, dim=0)

        return mapped_labels
