import logging
from typing import Any, Dict, Mapping, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch import nn
from torchmetrics import Accuracy

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from fs_grl.data.datamodule import MetaData
from fs_grl.data.episode import EpisodeBatch
from fs_grl.pl_modules.pl_module import MyLightningModule

pylogger = logging.getLogger(__name__)


class DMLBaseline(MyLightningModule):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # We want to skip metadata since it is saved separately by the
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata

        self.model = instantiate(
            self.hparams.model,
            cfg=self.hparams.model,
            feature_dim=self.metadata.feature_dim,
            num_classes=self.metadata.num_classes,
            num_classes_per_episode=self.metadata.num_classes_per_episode,
            _recursive_=False,
        )

        self.test_metrics = nn.ModuleDict({"test/micro_acc": Accuracy(num_classes=metadata.num_classes_per_episode)})
        self.val_metrics = nn.ModuleDict({"val/micro_acc": Accuracy(num_classes=metadata.num_classes_per_episode)})

    def forward(self, batch: EpisodeBatch) -> Dict:
        """Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """

        similarities = self.model(batch)

        loss = self.model.loss_func(similarities, batch.cosine_targets)
        return {"loss": loss, "similarities": similarities.detach()}

    def step(self, batch, split: str) -> Mapping[str, Any]:

        model_out = self(batch)
        loss = model_out["loss"]
        self.log_dict({f"loss/{split}": loss}, on_epoch=True, on_step=True)

        return model_out

    def training_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch, "train")
        return step_out

    def validation_step(self, batch: EpisodeBatch, batch_idx: int):
        step_out = self.step(batch, "val")

        # shape ~(num_episodes * num_queries_per_class * num_classes_per_episode)
        similarities = step_out["similarities"]

        num_classes_per_episode = batch.episode_hparams.num_classes_per_episode
        reshaped_similarities = similarities.reshape((-1, num_classes_per_episode))
        pred_labels = torch.argmax(reshaped_similarities, dim=-1)

        target_labels = batch.local_labels

        for metric_name, metric in self.val_metrics.items():
            metric_res = metric(preds=pred_labels, target=target_labels)
            self.log(name=metric_name, value=metric_res, on_step=True, on_epoch=True)

        return step_out

    def test_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch, "test")

        # shape ~(num_episodes * num_queries_per_class * num_classes_per_episode)
        similarities = step_out["similarities"]

        num_classes_per_episode = batch.episode_hparams.num_classes_per_episode
        reshaped_similarities = similarities.reshape((-1, num_classes_per_episode))
        pred_labels = torch.argmax(reshaped_similarities, dim=-1)

        target_labels = batch.local_labels

        for metric_name, metric in self.test_metrics.items():
            metric_res = metric(preds=pred_labels, target=target_labels)
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
