from typing import Any, Dict, Optional

import torch

from fs_grl.data.datamodule.metadata import MetaData
from fs_grl.pl_modules.base_module import BaseModule


class MetaLearningModel(BaseModule):
    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:

        super().__init__(metadata)
        self.save_hyperparameters(logger=False, ignore=("metadata",))
        self.automatic_optimization = False

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def step(self, metatrain: bool, batch: Any):
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int):
        """
        A meta-training step

        :param batch:
        :param batch_idx:
        :return
        """

        outer_loss, inner_loss, outer_acc, inner_acc = self.step(batch=batch, metatrain=True)

        self.log_dict(
            {"metatrain/inner_loss": inner_loss.item(), "metatrain/inner_accuracy": inner_acc.compute()},
            on_epoch=False,
            on_step=True,
            prog_bar=False,
        )
        self.log_dict(
            {"metatrain/outer_loss": outer_loss.item(), "metatrain/outer_accuracy": outer_acc.compute()},
            on_epoch=False,
            on_step=True,
            prog_bar=True,
        )

    def validation_step(self, batch: Any, batch_idx: int):
        """
        A meta-validation step

        :param batch:
        :param batch_idx:
        :return
        """

        # force training
        torch.set_grad_enabled(True)
        self.gnn_mlp.train()

        outer_loss, inner_loss, outer_acc, inner_acc = self.step(batch=batch, metatrain=False)

        self.log_dict(
            {"metaval/inner_loss": inner_loss.item(), "metaval/inner_accuracy": inner_acc.compute()}, prog_bar=False
        )
        self.log_dict(
            {"metaval/outer_loss": outer_loss.item(), "metaval/outer_accuracy": outer_acc.compute()}, prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        """
        A meta-testing step

        :param batch:
        :param batch_idx:
        :return
        """

        # force training
        torch.set_grad_enabled(True)
        self.gnn_mlp.train()

        outer_loss, inner_loss, outer_acc, inner_acc = self.step(batch=batch, metatrain=False)

        self.log_dict(
            {
                "metatest/outer_loss": outer_loss.item(),
                "metatest/inner_loss": inner_loss.item(),
                "metatest/inner_accuracy": inner_acc.compute(),
                "metatest/outer_accuracy": outer_acc.compute(),
            },
            on_step=True,
        )
