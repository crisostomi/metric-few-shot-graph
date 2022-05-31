import logging
from typing import Any, Mapping, Optional

import torch
from hydra.utils import instantiate

from nn_core.model_logging import NNLogger

from fs_grl.data.datamodule.metadata import MetaData
from fs_grl.data.episode.episode_batch import EpisodeBatch
from fs_grl.pl_modules.base_module import BaseModule

pylogger = logging.getLogger(__name__)


class DistanceMetricLearning(BaseModule):
    logger: NNLogger

    def __init__(
        self,
        metadata: Optional[MetaData] = None,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(metadata)

        self.save_hyperparameters(logger=False, ignore=("metadata",))

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

    def forward(self, batch: EpisodeBatch) -> torch.Tensor:
        """
        :return similarities, tensor ~ (B*(N*Q)*N) containing for each episode the similarity
                between each of the N*Q queries and the N label prototypes
        """

        model_out = self.model(batch)

        return model_out

    def step(self, batch, split: str) -> Mapping[str, Any]:

        model_out = self(batch)

        losses = self.model.compute_losses(model_out, batch)

        self.log_losses(losses, split)

        return {"model_out": model_out, "losses": losses, "loss": losses["total"]}

    def training_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch, "train")

        predictions = self.model.get_predictions(step_out, batch)

        for metric_name, metric in self.train_metrics.items():
            metric_res = metric(preds=predictions, target=batch.queries.y)
            self.log(name=metric_name, value=metric_res, on_step=True, on_epoch=True)

        return step_out

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
