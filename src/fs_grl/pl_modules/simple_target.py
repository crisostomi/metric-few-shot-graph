import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import hydra
import pytorch_lightning
import torch
from hydra.utils import instantiate
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from fs_grl.data.datamodule.metadata import MetaData
from fs_grl.data.episode.episode_batch import EpisodeBatch
from fs_grl.pl_modules.transfer_learning_target import TransferLearningTarget

pylogger = logging.getLogger(__name__)


class SimpleTarget(TransferLearningTarget):
    def __init__(
        self,
        embedder: nn.Module,
        initial_state_path: str,
        num_finetuning_steps: int,
        metadata: Optional[MetaData] = None,
        *args,
        **kwargs,
    ):
        self.metadata = metadata

        super().__init__(embedder=embedder, initial_state_path=initial_state_path, metadata=metadata)

        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.classes = metadata.classes_split["novel"]

        self.log_prefix = "meta-testing"
        self.num_finetuning_steps = num_finetuning_steps

        self.embedder = embedder
        self.freeze_embedder()

        self.classifier = instantiate(self.hparams.classifier, output_dim=len(self.classes))

        self.loss_func = nn.CrossEntropyLoss()

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
        self.train()
        self.embedder.eval()

        optimizer = self.optimizers()

        total_loss = torch.tensor(0.0)
        for i in range(self.num_finetuning_steps):
            samples = batch.supports
            embeddings = self.embedder(samples)
            logits = self.classifier(embeddings)

            loss = self.loss_func(logits, samples.y)

            self.manual_backward(loss)
            optimizer.step()
            self.zero_grad()

            total_loss += loss.cpu()

        return total_loss

    def on_train_batch_start(self, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        self.reset_fine_tuning()
        self.freeze_embedder()

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        """
        Testing of the fine-tuning phase. After the model has trained on the K*N supports, it is
        tested on the N*Q queries
        """
        self.eval()
        logits = self(batch.queries)["logits"]

        class_probs = torch.log_softmax(logits, dim=-1)
        preds = torch.argmax(class_probs, dim=-1)

        for metric in self.test_metrics.values():
            metric(preds=preds, target=batch.queries.y)

        self.log_metrics(split="test", on_step=True, on_epoch=True, cm_reset=False)

    def reset_fine_tuning(self):
        """
        Resets the model to the original pretrained state
        """
        self.load_state_dict(torch.load(self.initial_state_path))
        self.trainer: pytorch_lightning.Trainer
        (
            self.trainer.optimizers,
            self.trainer.lr_schedulers,
            self.trainer.optimizer_frequencies,
        ) = self.trainer.init_optimizers(model=None)

    def freeze_embedder(self):
        self.embedder.eval()
        self.embedder.requires_grad_(False)

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

        schedulers = []
        if "lr_scheduler" not in self.hparams:
            lr_scheduler_config = {
                "scheduler": LambdaLR(optimizer=opt, lr_lambda=lambda _: 1),
                "name": f"meta-testing/lr-{opt.__class__.__name__}",
            }

            schedulers.append(lr_scheduler_config)
        else:
            schedulers.append(hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt))

        return [opt], schedulers
