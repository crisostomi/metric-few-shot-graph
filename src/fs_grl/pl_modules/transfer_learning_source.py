import logging
from typing import Any, Mapping, Optional

import torch

from nn_core.model_logging import NNLogger

from fs_grl.data.datamodule import MetaData
from fs_grl.pl_modules.transfer_learning import TransferLearningBaseline

pylogger = logging.getLogger(__name__)


class TransferLearningSource(TransferLearningBaseline):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__(metadata)

        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.classes = self.metadata.classes_split["base"]

        self.embedder = None
        self.classifier = None
        self.loss_func = None

    def step(self, batch, split: str) -> Mapping[str, Any]:

        model_out = self(batch)

        losses = self.compute_losses(model_out, batch)

        self.log_losses(losses, split)

        return {"model_out": model_out, "losses": losses, "loss": losses["total"]}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch, "train")

        model_out = step_out["model_out"]

        logits = model_out["logits"]

        class_probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(class_probs, dim=-1)

        for metric_name, metric in self.train_metrics.items():
            metric_res = metric(preds=preds, target=batch.y)
            self.log(name=metric_name, value=metric_res, on_step=True, on_epoch=True)

        return step_out

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch, "val")

        model_out = step_out["model_out"]

        logits = model_out["logits"]

        class_probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(class_probs, dim=-1)

        for metric_name, metric in self.val_metrics.items():
            metric_res = metric(preds=preds, target=batch.y)
            if "none" not in metric_name and "cm" not in metric_name:
                self.log(name=metric_name, value=metric_res, on_step=True, on_epoch=True)

        return step_out
