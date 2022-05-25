import logging
from abc import ABC

from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from fs_grl.pl_modules.pl_module import MyLightningModule

pylogger = logging.getLogger(__name__)


class TransferLearningBaseline(MyLightningModule, ABC):
    def __init__(self, metadata):
        self.metadata = metadata
        super().__init__()

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log_metrics(split="val", on_step=False, on_epoch=True, cm_reset=True)

    def compute_total_loss(self, losses):
        return sum(
            [
                loss_value * self.loss_weights[loss_name]
                for loss_name, loss_value in losses.items()
                if loss_name != "total"
            ]
        )
