import logging
from typing import Any, Dict, Mapping, Optional

import torch
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torchmetrics import FBeta

from fs_grl.data.datamodule import MetaData
from fs_grl.data.episode import EpisodeBatch
from fs_grl.modules.mlp import MLP
from fs_grl.pl_modules.transfer_learning import TransferLearningBaseline

pylogger = logging.getLogger(__name__)


class TransferLearningTarget(TransferLearningBaseline):
    def __init__(
        self,
        embedder,
        initial_state_path,
        classifier_num_mlp_layers,
        metadata: Optional[MetaData] = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=("metadata",))
        self.metadata = metadata

        self.embedder = embedder
        self.freeze_embedder()

        self.initial_state_path = initial_state_path
        self.classifier_num_mlp_layers = classifier_num_mlp_layers

        self.classes = metadata.classes_split["novel"]
        self.log_prefix = "meta-testing"

        reductions = ["micro", "weighted", "macro"]  # TODO: add None
        metrics = ["F1", "acc"]

        self.train_metrics = nn.ModuleDict(
            {
                f"{self.log_prefix}/train/{metric}/{reduction}": FBeta(num_classes=len(self.classes), average=reduction)
                for reduction in reductions
                for metric in metrics
            }
        )

        self.test_metrics = nn.ModuleDict(
            {
                f"{self.log_prefix}/test/{metric}/{reduction}": FBeta(num_classes=len(self.classes), average=reduction)
                for reduction in reductions
                for metric in metrics
            }
        )

        self.test_metrics[f"{self.log_prefix}/test/cm"] = torchmetrics.ConfusionMatrix(
            num_classes=len(self.classes), normalize=None
        )

        self.classifier = MLP(
            num_layers=self.classifier_num_mlp_layers,
            input_dim=self.embedder.embedding_dim,
            output_dim=len(self.classes),
            hidden_dim=self.embedder.embedding_dim // 2,
        )

        self.loss_func = nn.CrossEntropyLoss()

        self.initial_state_path = initial_state_path
        self.save_initial_state()

    def save_initial_state(self):
        torch.save(self.state_dict(), self.initial_state_path)

    def forward(self, samples) -> Dict:
        """
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """

        embeddings = self.embedder(samples)
        logits = self.classifier(embeddings)

        return {"logits": logits}

    def step(self, samples, split: str) -> Mapping[str, Any]:

        model_out = self(samples)

        loss = self.loss_func(model_out["logits"], samples.y)

        model_out["loss"] = loss
        self.log_dict({f"{self.log_prefix}/loss/{split}": loss}, on_epoch=True, on_step=True)

        return model_out

    def training_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch.supports, "train")

        return step_out

    def on_train_batch_start(self, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        self.reset_fine_tuning()
        self.freeze_embedder()

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        logits = self(batch.queries)["logits"]

        class_probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(class_probs, dim=-1)

        for metric in self.test_metrics.values():
            metric(preds=preds, target=batch.queries.y)

        self.log_metrics(split="test", on_step=True, on_epoch=True)

    def reset_fine_tuning(self):
        self.load_state_dict(torch.load(self.initial_state_path))

    def freeze_embedder(self):
        """ """
        self.embedder.eval()
        for param in self.embedder.parameters():
            param.requires_grad = False
