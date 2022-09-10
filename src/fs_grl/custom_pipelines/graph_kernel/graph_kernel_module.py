import logging
import time
from typing import Any, Mapping, Optional

import torch
from hydra.utils import instantiate

from nn_core.model_logging import NNLogger

from fs_grl.custom_pipelines.graph_kernel.utils import batch_to_grakel
from fs_grl.data.datamodule.metadata import MetaData
from fs_grl.data.episode.episode_batch import EpisodeBatch
from fs_grl.pl_modules.base_module import BaseModule

pylogger = logging.getLogger(__name__)


class GraphKernel(BaseModule):
    logger: NNLogger

    def __init__(
        self,
        metadata: Optional[MetaData] = None,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(metadata)

        self.save_hyperparameters(logger=False, ignore=("metadata",))
        self.classes = metadata.classes_split["novel"]

    def forward(self, batch: EpisodeBatch) -> torch.Tensor:
        """
        :return similarities, tensor ~ (B*(N*Q)*N) containing for each episode the similarity
                between each of the N*Q queries and the N label prototypes
        """
        ts = time.time()
        kernel = instantiate(self.hparams.graph_kernel, _recursive_=False)
        print(f"Instantiate kernel: {time.time() - ts}")
        clf = instantiate(self.hparams.classifier, _recursive_=False)
        print(f"Instantiate classifier: {time.time() - ts}")

        supports = batch_to_grakel(batch.supports, self.hparams.node_labels_tag)
        supports_y = batch.supports.y.numpy()
        queries = batch_to_grakel(batch.queries, self.hparams.node_labels_tag)
        print(f"Dataloader to grakel: {time.time() - ts}")

        K_sup = kernel.fit_transform(supports)
        print(f"Fit kernel: {time.time() - ts}")

        # unique_supports_y, _ = np.unique(supports_y, return_index=True)

        # prototypes = []
        # for cls in unique_supports_y:
        #     indices = supports_y == cls
        #     prototype = K_sup[indices].mean(axis=0)
        #     prototypes.append(prototype)

        # prototypes = np.array(prototypes)

        clf.fit(K_sup, supports_y)
        print(f"Fit classifier: {time.time() - ts}")

        K_qu = kernel.transform(queries)
        print(f"Kernel queries: {time.time() - ts}")
        y_pred = clf.predict(K_qu)
        print(f"Query predictions: {time.time() - ts}")

        return torch.from_numpy(y_pred)

    def training_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:
        pass

    def validation_step(self, batch: EpisodeBatch, batch_idx: int):
        pass

    def test_step(self, batch: EpisodeBatch, batch_idx: int) -> Mapping[str, Any]:

        predictions = self(batch)

        for metric_name, metric in self.test_metrics.items():
            metric(preds=predictions, target=batch.queries.y)

        self.log_metrics(split="test", on_step=True, on_epoch=True, cm_reset=False)

        return predictions

    def configure_optimizers(self):
        pass
