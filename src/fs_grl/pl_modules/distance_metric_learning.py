import logging
from typing import Any, Mapping, Optional

import torch
from hydra.utils import instantiate

from nn_core.model_logging import NNLogger

from fs_grl.data.datamodule.metadata import MetaData
from fs_grl.data.episode.episode_batch import EpisodeBatch
from fs_grl.modules.architectures.tadam import TADAM
from fs_grl.pl_modules.base_module import BaseModule
from fs_grl.pl_modules.utils import log_tsne_plot, prototypes_dict_to_tensor

pylogger = logging.getLogger(__name__)


class DistanceMetricLearning(BaseModule):
    logger: NNLogger

    def __init__(
        self,
        metadata: Optional[MetaData] = None,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(metadata)

        self.save_hyperparameters(logger=False, ignore=("metadata",))

        # classes should be sorted, might add an assert later
        self.classes = list(metadata.classes_to_label_dict.keys())
        self.label_to_class_dict = {v: k for k, v in metadata.classes_to_label_dict.items()}

        self.model = instantiate(
            self.hparams.model,
            cfg=self.hparams.model,
            feature_dim=self.metadata.feature_dim,
            num_classes=self.metadata.num_classes,
            num_classes_per_episode=self.metadata.num_classes_per_episode,
            _recursive_=False,
        )

    def forward(self, batch: EpisodeBatch) -> torch.Tensor:
        """
        :return similarities, tensor ~ (B*(N*Q)*N) containing for each episode the similarity
                between each of the N*Q queries and the N label prototypes
        """

        model_out = self.model(batch)

        return model_out

    def step(self, batch, split: str, batch_idx: int = None) -> Mapping[str, Any]:

        model_out = self(batch)

        if batch_idx is not None and batch_idx < 10:

            query_embeds = model_out["embedded_queries"].detach().cpu()
            query_labels = batch.queries.y.detach().cpu()

            support_embeds = model_out["embedded_supports"].detach().cpu()
            support_labels = batch.supports.y.detach().cpu()

            prototype_embeds, prototype_labels = prototypes_dict_to_tensor(model_out["prototypes_dicts"][0])

            embeds = torch.cat([support_embeds, query_embeds, prototype_embeds], dim=0)
            classes = torch.cat([support_labels, query_labels, prototype_labels], dim=0)
            lens = {"support": len(support_labels), "query": len(query_labels), "prototype": len(prototype_labels)}

            log_tsne_plot(embeds, classes, lens, batch_idx, self.model, self.hparams, self.logger)

        losses = self.model.compute_losses(model_out, batch)

        self.log_losses(losses, split)

        return {"model_out": model_out, "losses": losses, "loss": losses["total"]}

    def training_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch, "train")

        predictions = self.model.get_predictions(step_out, batch)

        for metric_name, metric in self.train_metrics.items():
            metric_res = metric(preds=predictions, target=batch.queries.y)
            self.log(name=metric_name, value=metric_res, on_step=True, on_epoch=True)

        if isinstance(self.model, TADAM):
            for conv_ind, conv in enumerate(self.model.embedder.node_embedder.convs):
                self.log(f"TAE_multipliers/gamma_0_{conv_ind}", self.model.task_embedding_network.gamma_0[conv_ind])
                self.log(f"TAE_multipliers/beta_0_{conv_ind}", self.model.task_embedding_network.beta_0[conv_ind])

        return step_out

    def validation_step(self, batch: EpisodeBatch, batch_idx: int):
        step_out = self.step(batch, "val")

        predictions = self.model.get_predictions(step_out, batch)

        for metric_name, metric in self.val_metrics.items():
            metric(preds=predictions, target=batch.queries.y)

        self.log_metrics(split="val", on_step=True, on_epoch=True, cm_reset=False)

        return step_out

    def test_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch, "test", batch_idx)

        predictions = self.model.get_predictions(step_out, batch)

        for metric_name, metric in self.test_metrics.items():
            metric(preds=predictions, target=batch.queries.y)

        self.log_metrics(split="test", on_step=True, on_epoch=True, cm_reset=False)

        return step_out
