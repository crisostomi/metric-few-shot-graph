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

pylogger = logging.getLogger(__name__)


class DistanceMetricLearning(MyLightningModule):
    logger: NNLogger

    def __init__(
        self,
        # train_data_list_by_label,
        artificial_regularizer_weight=0.5,
        intra_class_variance_weight=0.0,
        metadata: Optional[MetaData] = None,
        *args,
        **kwargs,
    ) -> None:
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
        self.artificial_regularizer_weight = artificial_regularizer_weight
        self.intra_class_variance_weight = intra_class_variance_weight

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
        # self.train_data_list_by_label = train_data_list_by_label

    def forward(self, batch: EpisodeBatch) -> torch.Tensor:
        """
        :return similarities, tensor ~ (B*(N*Q)*N) containing for each episode the similarity
                between each of the N*Q queries and the N label prototypes
        """

        model_out = self.model(batch)

        return model_out

    def step(self, batch, split: str) -> Mapping[str, Any]:

        model_out = self(batch)

        regularizer_term = 0
        if self.artificial_regularizer_weight > 0:
            regularizer_term = self.model.compute_crossover_regularizer(model_out, batch)
            self.log_dict({f"loss/{split}/artificial_regularizer": regularizer_term}, on_epoch=True, on_step=True)

        intra_class_variance_term = 0
        if self.intra_class_variance_weight > 0:
            intra_class_variance_term = self.model.get_intra_class_variance(
                model_out["embedded_supports"], model_out["class_prototypes"], batch
            )
            self.log_dict(
                {f"loss/{split}/intra_class_variance": intra_class_variance_term}, on_epoch=True, on_step=True
            )

        margin_loss = self.model.compute_loss(model_out, batch)

        loss = (
            margin_loss
            + self.artificial_regularizer_weight * regularizer_term
            + self.intra_class_variance_weight * intra_class_variance_term
        )

        self.log_dict({f"loss/{split}": loss}, on_epoch=True, on_step=True)

        return {"model_out": model_out, "loss": loss}

    def training_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch, "train")

        predictions = self.model.get_predictions(step_out, batch)

        for metric_name, metric in self.train_metrics.items():
            metric_res = metric(preds=predictions, target=batch.queries.y)
            self.log(name=metric_name, value=metric_res, on_step=True, on_epoch=True)

        return step_out

    # def on_train_end(self) -> None:
    #     base_prototypes = compute_global_prototypes(self, self.train_data_list_by_label, self.label_to_class_dict)
    #     self.base_prototypes = base_prototypes

    def validation_step(self, batch: EpisodeBatch, batch_idx: int):
        step_out = self.step(batch, "val")

        predictions = self.model.get_predictions(step_out, batch)

        for metric_name, metric in self.val_metrics.items():
            metric(preds=predictions, target=batch.queries.y)

        self.log_metrics(split="val", on_step=True, on_epoch=True, cm_reset=False)

        return step_out

    def test_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch, "test")

        predictions = self.model.get_predictions(step_out, batch)

        for metric_name, metric in self.test_metrics.items():
            metric(preds=predictions, target=batch.queries.y)

        self.log_metrics(split="test", on_step=True, on_epoch=True, cm_reset=False)

        return step_out
