import itertools
import logging
from typing import Any, Dict, Mapping, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torchmetrics
from hydra.utils import instantiate
from torch import nn
from torchmetrics import Accuracy, FBetaScore

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from fs_grl.data.datamodule import MetaData
from fs_grl.modules.mlp import MLP
from fs_grl.pl_modules.transfer_learning import TransferLearningBaseline

pylogger = logging.getLogger(__name__)


class TransferLearningSource(TransferLearningBaseline):
    logger: NNLogger

    def __init__(self, classifier_num_mlp_layers, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # We want to skip metadata since it is saved separately
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata
        self.classifier_num_mlp_layers = classifier_num_mlp_layers

        self.embedder = instantiate(
            self.hparams.embedder,
            feature_dim=self.metadata.feature_dim,
            _recursive_=False,
        )

        self.classes = metadata.classes_split["base"]

        reductions = ["micro", "weighted", "macro", "none"]
        metrics = (("F1", FBetaScore), ("acc", Accuracy))

        self.val_metrics = nn.ModuleDict(
            {
                f"val/{metric_name}/{reduction}": metric(num_classes=len(self.classes), average=reduction)
                for reduction, (metric_name, metric) in itertools.product(reductions, metrics)
            }
        )
        self.val_metrics["val/cm"] = torchmetrics.ConfusionMatrix(num_classes=len(self.classes), normalize=None)
        self.train_metrics = nn.ModuleDict({"train/acc/micro": Accuracy(num_classes=len(self.classes))})

        self.classifier = MLP(
            num_layers=self.classifier_num_mlp_layers,
            input_dim=self.embedder.embedding_dim,
            output_dim=len(self.classes),
            hidden_dim=self.embedder.embedding_dim // 2,
        )

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, batch: Any) -> Dict:
        """
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """

        embeddings = self.embedder(batch)
        logits = self.classifier(embeddings)

        loss = self.loss_func(logits, batch.y)
        return {"loss": loss, "logits": logits}

    def step(self, batch, split: str) -> Mapping[str, Any]:

        model_out = self(batch)
        loss = model_out["loss"]
        self.log_dict({f"loss/{split}": loss}, on_epoch=True, on_step=True)

        return model_out

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch, "train")

        logits = step_out["logits"]

        class_probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(class_probs, dim=-1)

        for metric_name, metric in self.train_metrics.items():
            metric_res = metric(preds=preds, target=batch.y)
            self.log(name=metric_name, value=metric_res, on_step=True, on_epoch=True)

        return step_out

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch, "val")

        logits = step_out["logits"]

        class_probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(class_probs, dim=-1)

        for metric_name, metric in self.val_metrics.items():
            metric_res = metric(preds=preds, target=batch.y)
            if "none" not in metric_name and "cm" not in metric_name:
                self.log(name=metric_name, value=metric_res, on_step=True, on_epoch=True)

        return step_out


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Lightning Module.
    Args:
        cfg: the hydra configuration
    """
    _: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
