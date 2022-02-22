import logging
from typing import Any, Mapping, Optional

import torch
from torch.nn import NLLLoss

from nn_core.model_logging import NNLogger

from fs_grl.data.datamodule import MetaData
from fs_grl.data.episode import EpisodeBatch
from fs_grl.pl_modules.distance_metric_learning import DistanceMetricLearning

pylogger = logging.getLogger(__name__)


class GraphProtoNet(DistanceMetricLearning):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__(metadata)

        self.save_hyperparameters(logger=False, ignore=("metadata",))
        self.loss_func = NLLLoss()

    def forward(self, batch: EpisodeBatch) -> torch.Tensor:
        """
        :return similarities, tensor ~ (B*(N*Q)*N) containing for each episode the similarity
                between each of the N*Q queries and the N label prototypes
        """

        similarities = self.model(batch)

        return similarities

    def step(self, batch: EpisodeBatch, split: str) -> Mapping[str, Any]:

        # shape (B * N*Q * N)
        similarities = self(batch)

        # shape (B, N*Q, N)
        similarities = similarities.reshape(batch.num_episodes, -1, batch.episode_hparams.num_classes_per_episode)
        probabilities = torch.log_softmax(similarities, dim=-1)

        labels_per_episode = batch.local_labels.reshape(batch.num_episodes, -1)

        cum_loss = 0
        for episode in range(batch.num_episodes):
            cum_loss += self.loss_func(probabilities[episode], labels_per_episode[episode])

        cum_loss /= batch.num_episodes
        self.log_dict({f"loss/{split}": cum_loss}, on_epoch=True, on_step=True)

        return {"similarities": similarities, "loss": cum_loss}

    def training_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch, "train")

        similarities = step_out["similarities"]

        predictions = self.get_predictions(similarities, batch)
        for metric_name, metric in self.train_metrics.items():
            metric_res = metric(preds=predictions, target=batch.queries.y)
            self.log(name=metric_name, value=metric_res, on_step=True, on_epoch=True)

        return step_out
