import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torchmetrics
from hydra.utils import instantiate
from torch import nn
from torch.optim import Optimizer
from torchmetrics import Accuracy

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from fs_grl.data.datamodule import MetaData
from fs_grl.data.episode import EpisodeBatch

pylogger = logging.getLogger(__name__)


class TransferLearningTarget(pl.LightningModule):
    def __init__(self, embedding_dim, num_classes, metadata: Optional[MetaData] = None, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=("metadata",))
        self.metadata = metadata

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.embedder = instantiate(
            self.hparams.embedder,
            cfg=self.hparams.embedder,
            feature_dim=self.metadata.feature_dim,
            num_classes=self.metadata.num_classes,
            episode_hparams=self.metadata.episode_hparams,
            _recursive_=False,
        )

        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, batch) -> Dict:
        """
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """

        embeddings = self.embedder(batch)
        logits = self.classifier(embeddings)

        loss = self.loss_func(logits, batch.label_targets)

        return {"loss": loss}

    def step(self, batch, split: str) -> Mapping[str, Any]:

        model_out = self(batch)
        loss = model_out["loss"]
        self.log_dict({f"loss/{split}": loss}, on_epoch=True, on_step=True)

        return model_out

    def training_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch, "train")
        return step_out

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters(), _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]
