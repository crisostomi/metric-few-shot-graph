import logging
from typing import Any, Dict, Optional

from hydra.utils import instantiate
from torch import nn
from torch_geometric.data import Batch

from nn_core.model_logging import NNLogger

from fs_grl.data.datamodule import MetaData
from fs_grl.modules.graph_embedder import GraphEmbedder
from fs_grl.pl_modules.transfer_learning_source import TransferLearningSource

pylogger = logging.getLogger(__name__)


class SimpleSource(TransferLearningSource):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__(metadata)

        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.embedder: GraphEmbedder = instantiate(
            self.hparams.embedder, cfg=self.hparams.embedder, feature_dim=self.metadata.feature_dim, _recursive_=False
        )

        self.classifier = instantiate(self.hparams.classifier, output_dim=len(self.classes))

        self.loss_func = nn.CrossEntropyLoss()

        self.loss_weights = {"classification_loss": 1}

    def forward(self, batch: Any) -> Dict:
        """
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """

        embeddings = self.embedder(batch)
        logits = self.classifier(embeddings)

        return {"logits": logits}

    def compute_losses(self, model_out: Dict, batch: Batch):

        logits = model_out["logits"]
        targets = batch.y

        losses = {"classification_loss": self.loss_func(logits, targets)}

        losses["total"] = self.compute_total_loss(losses)

        return losses
