import logging
from typing import List

import hydra
import numpy as np
import plotly.graph_objects as go
import torch
from omegaconf import ListConfig
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader

pylogger = logging.getLogger(__name__)


class TSNEPlot(Callback):
    def __init__(self, num_samples) -> None:
        super().__init__()
        self.num_samples = num_samples

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dataset = trainer.datamodule.split_base_novel_samples()["novel"]
        dataset = self.sample_examples_tsne(dataset)

        dataloader = DataLoader(dataset)
        embedder = pl_module.model.embedder

        queries_embeddings = []
        classes = []
        for batch in dataloader:
            batch.to(pl_module.device)
            embedded_queries = embedder(batch)

            queries_embeddings.append(embedded_queries.cpu())
            classes.append(batch.y.cpu())

        queries_embeddings = torch.cat(queries_embeddings, dim=0)
        classes = torch.cat(classes, dim=0)
        assert queries_embeddings.size(0) == classes.size(0)

        tsne_results = self.compute_tsne(embeddings=queries_embeddings)
        self.plot_and_log_tsne(tsne_results=tsne_results, labels=classes, trainer=trainer)

    def sample_examples_tsne(self, dataset):
        idxs = np.arange(len(dataset))
        np.random.shuffle(idxs)
        dataset = [dataset[idx] for idx in idxs[: self.num_samples]]
        return dataset

    def compute_tsne(self, embeddings):
        tsne = TSNE(n_iter=1000, verbose=1)
        tsne_results = tsne.fit_transform(embeddings)

        return tsne_results

    def plot_and_log_tsne(self, tsne_results, labels, trainer):
        data = []
        for label in labels.unique():
            indices = torch.where(labels == label)[0]
            data.append(
                go.Scatter(
                    x=tsne_results[indices, 0], y=tsne_results[indices, 1], mode="markers", name=f"Class {label+1}"
                )
            )
        plot = go.Figure(data=data)

        trainer.logger.experiment.log({"T-SNE": plot})


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


def get_checkpoint_callback(callbacks):
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback


def handle_fast_dev_run(cfg):
    pylogger.info(f"Debug mode <{cfg.train.trainer.fast_dev_run=}>. Forcing debugger friendly configuration!")
    # Debuggers don't like GPUs nor multiprocessing
    cfg.train.trainer.gpus = 0
    cfg.nn.data.num_workers.train = 0
    cfg.nn.data.num_workers.val = 0
    cfg.nn.data.num_workers.test = 0
