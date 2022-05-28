from typing import Any, Mapping, Optional

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

from fs_grl.data.datamodule import MetaData
from fs_grl.data.episode import EpisodeBatch
from fs_grl.gsm.modules import ClassifierLayer
from fs_grl.pl_modules.transfer_learning_target import TransferLearningTarget


class GraphSpectralMeasuresTarget(TransferLearningTarget):
    def __init__(
        self,
        embedder: nn.Module,
        initial_state_path: str,
        metadata: Optional[MetaData] = None,
        *args,
        **kwargs,
    ):

        super().__init__(embedder=embedder, initial_state_path=initial_state_path, metadata=metadata)

        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.classes = metadata.classes_split["novel"]

        self.log_prefix = "meta-testing"

        self.embedder = embedder
        self.freeze_embedder()

        num_classes = len(self.classes)
        self.classifier = ClassifierLayer(
            "linear", final_gat_out_dim=self.embedder.final_gat_out_dim, num_classes=num_classes
        )

        self.loss_func = nn.CrossEntropyLoss()
        self.save_initial_state()

        self.episodes_counter = 0

    def save_initial_state(self):
        torch.save(self.state_dict(), self.initial_state_path)

    def training_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:
        self.train()
        self.embedder.eval()

        supports_queries = batch.supports.to_data_list() + batch.queries.to_data_list()

        output_embeds, (node_embeds, Adj_block_idx), gin_preds, edges = self.embedder(
            supports_queries, is_metatest=True
        )

        logits, _ = self.classifier(output_embeds[: batch.num_supports_per_episode])

        targets = batch.supports.y

        loss = self.loss_func(logits, targets)

        self.episodes_counter += 1

        return {"logits": logits, "targets": targets, "loss": loss}

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        """
        Testing of the fine-tuning phase. After the model has trained on the K*N supports, it is
        tested on the N*Q queries
        """

        self.eval()

        supports = batch.supports.to_data_list()
        queries = batch.queries.to_data_list()

        num_supports = batch.num_supports_per_episode

        supports_and_queries = supports + queries

        output_embeds, (node_embeds, Adj_block_idx), gin_preds, edges = self.embedder(
            supports_and_queries, is_metatest=True
        )

        logits, embeddings = self.classifier(output_embeds)

        logits = logits[num_supports:]
        targets = batch.queries.y

        class_probs = torch.log_softmax(logits, dim=-1)
        preds = torch.argmax(class_probs, dim=-1)

        for metric in self.test_metrics.values():
            metric(preds=preds, target=targets)

        self.log_metrics(split="test", on_step=True, on_epoch=True, cm_reset=False)
