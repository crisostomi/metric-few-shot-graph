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


class MyLightningModule(pl.LightningModule):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata

        metric = torchmetrics.Accuracy()
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()

        self.model = instantiate(
            self.hparams.model,
            cfg=self.hparams.model,
            feature_dim=self.metadata.feature_dim,
            num_classes=self.metadata.num_classes,
            episode_hparams=self.metadata.episode_hparams,
            _recursive_=False,
        )

        self.val_metrics = nn.ModuleDict(
            {"val/micro_acc": Accuracy(num_classes=metadata.episode_hparams.num_classes_per_episode)}
        )

    def forward(self, batch: EpisodeBatch) -> Dict:
        """Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """

        similarities = self.model(batch)

        loss = self.model.loss_func(similarities, batch.cosine_targets)
        return {"loss": loss, "similarities": similarities}

    def step(self, batch, split: str) -> Mapping[str, Any]:

        model_out = self(batch)
        loss = model_out["loss"]
        self.log_dict({f"loss/{split}": loss}, on_epoch=True, on_step=True)

        return model_out

    def training_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch, "train")
        return step_out

    def validation_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:
        step_out = self.step(batch, "val")

        # shape (num_queries_batch) = num_queries_per_class * num_classes_per_episode * batch_size * num_classes_per_episode

        similarities = step_out["similarities"]

        num_classes_per_episode = batch.episode_hparams.num_classes_per_episode
        reshaped_similarities = similarities.reshape((-1, num_classes_per_episode))
        pred_labels = torch.argmax(reshaped_similarities, dim=-1)

        target_labels = batch.label_targets

        for metric_name, metric in self.val_metrics.items():
            metric_res = metric(preds=pred_labels, target=target_labels)
            self.log(name=metric_name, value=metric_res, on_step=True, on_epoch=True)

        # print(pred_labels)
        return step_out

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        step_out = self.step(batch, "test")
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
