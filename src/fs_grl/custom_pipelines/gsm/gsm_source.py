import logging
from typing import Any, Dict, Optional

from hydra.utils import instantiate
from torch import nn
from torch_geometric.data import Batch

from nn_core.model_logging import NNLogger

from fs_grl.custom_pipelines.gsm.modules import ClassifierLayer
from fs_grl.data.datamodule.metadata import MetaData
from fs_grl.pl_modules.transfer_learning_source import TransferLearningSource

pylogger = logging.getLogger(__name__)


class GraphSpectralMeasuresSource(TransferLearningSource):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__(metadata)

        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.embedder = instantiate(
            self.hparams.embedder,
            model_cfg=self.hparams.embedder,
            feature_dim=self.metadata.feature_dim,
            _recursive_=False,
        )

        self.loss_func = nn.CrossEntropyLoss()
        self.super_class_loss_func = nn.CrossEntropyLoss()

        num_classes = len(self.classes)
        self.classifier = ClassifierLayer(
            "linear", final_gat_out_dim=self.embedder.final_gat_out_dim, num_classes=num_classes
        )

        self.loss_weights = {"classification_loss": 1, "superclass_loss": 1}

    def forward(self, batch: Any) -> Dict:
        """
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """

        data_list = batch.to_data_list()

        output_embeds, (node_embeds, Adj_block_idx), superclass_logits, edges = self.embedder(data_list)

        logits, _ = self.classifier(output_embeds)

        return {"logits": logits, "superclass_logits": superclass_logits}

    def compute_losses(self, model_out: Dict, batch: Batch):

        logits = model_out["logits"]
        targets = batch.y

        superclass_logits = model_out["superclass_logits"]
        superclass_targets = batch.super_class

        losses = {"classification_loss": 0, "superclass_loss": 0}

        losses["classification_loss"] = self.loss_func(logits, targets)
        losses["superclass_loss"] = self.super_class_loss_func(superclass_logits, superclass_targets)

        losses["total"] = self.compute_total_loss(losses)

        return losses

    def compute_total_loss(self, losses):
        return sum(
            [
                loss_value * self.loss_weights[loss_name]
                for loss_name, loss_value in losses.items()
                if loss_name != "total"
            ]
        )
