import logging
import time
from typing import List

import hydra
import numpy as np
import plotly.express as px
import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader

from fs_grl.data.utils import get_label_to_samples_map

pylogger = logging.getLogger(__name__)


class ClassesSimilarityCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.samples_per_class = 100

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        split_base_novel_samples = trainer.datamodule.split_base_novel_samples()
        train_dataset = split_base_novel_samples["base"]
        dataset = self.sample_data(train_dataset, self.samples_per_class)

        dataloader = DataLoader(dataset)
        embedder = pl_module.model.embedder
        embedder.eval()

        embeds_by_class = {}
        for batch in dataloader:
            batch.to(pl_module.device)

            embedded_sample = embedder(batch)
            for ind, cls in enumerate(batch.y.tolist()):
                embeds_by_class.setdefault(cls, []).append(embedded_sample[ind])

        prototypes_by_class = {}
        for cls, embeds in embeds_by_class.items():
            prototypes_by_class[str(cls)] = torch.mean(torch.stack(embeds), dim=0)

        prototypes_matrix = torch.stack(list(prototypes_by_class.values()))

        norm_prototypes = F.normalize(prototypes_matrix, dim=-1)
        cosine_similarity_matrix = norm_prototypes @ norm_prototypes.T

        hm = px.imshow(
            np.around(cosine_similarity_matrix.detach().cpu().numpy(), 3),
            x=list(prototypes_by_class.keys()),
            y=list(prototypes_by_class.keys()),
            text_auto=True,
            aspect="auto",
        )
        hm.update_xaxes(side="top")

        trainer.logger.experiment.log({"classes_similarity/heatmap": hm})

    def sample_data(self, dataset, num_samples):
        label_to_samples_map = get_label_to_samples_map(dataset)
        sampled_data = []
        for _, samples in label_to_samples_map.items():
            idxs = np.arange(len(samples))
            np.random.shuffle(idxs)
            upper_bound = min(len(samples), num_samples)
            sampled_data += [samples[idx] for idx in idxs[:upper_bound]]
        return sampled_data


def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.
    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated
    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


class LogTrainingTimeCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        trainer.logger.experiment.log({"training_time": time.time() - pl_module.train_start_time})


def get_checkpoint_callback(callbacks):
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback
