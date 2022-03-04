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
    def __init__(self, novel_samples, base_samples) -> None:
        super().__init__()
        self.novel_samples = novel_samples
        self.base_samples = base_samples

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        split_base_novel_samples = trainer.datamodule.split_base_novel_samples()
        base_dataset, novel_dataset = split_base_novel_samples["base"], split_base_novel_samples["novel"]
        novel_dataset = self.sample_examples_tsne(novel_dataset, self.novel_samples)
        base_dataset = self.sample_examples_tsne(base_dataset, self.base_samples)

        base_novel_dataset = base_dataset + novel_dataset

        dataloader = DataLoader(base_novel_dataset)
        embedder = pl_module.model.embedder
        embedder.eval()

        embeddings = []
        classes = []
        for batch in dataloader:
            batch.to(pl_module.device)
            embedded_sample = embedder(batch)

            embeddings.append(embedded_sample.cpu())
            classes.append(batch.y.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        classes = torch.cat(classes, dim=0)
        assert embeddings.size(0) == classes.size(0)

        tsne_results = self.compute_tsne(n_components=2, embeddings=embeddings)
        plot = self.tsne_plot(tsne_results=tsne_results, labels=classes)

        trainer.logger.experiment.log({"t-SNE": plot})

        tsne3d_results = self.compute_tsne(n_components=3, embeddings=embeddings)
        plot3d = self.tsne_plot(tsne_results=tsne3d_results, labels=classes)

        trainer.logger.experiment.log({"3d t-SNE": plot3d})

    def sample_examples_tsne(self, dataset, num_samples):
        idxs = np.arange(len(dataset))
        np.random.shuffle(idxs)
        dataset = [dataset[idx] for idx in idxs[:num_samples]]
        return dataset

    def compute_tsne(self, n_components, embeddings):
        tsne = TSNE(n_components=n_components, n_iter=1000, verbose=1)
        tsne_results = tsne.fit_transform(embeddings)

        return tsne_results

    def tsne_plot(self, tsne_results, labels):
        tsne_base = tsne_results[: self.base_samples, :]
        base_labels = labels[: self.base_samples]

        tsne_novel = tsne_results[-self.novel_samples :, :]
        novel_labels = labels[-self.novel_samples :]

        assert len(tsne_results) == (len(tsne_base) + len(tsne_novel))
        assert len(labels) == (len(base_labels) + len(novel_labels))

        base_scatter = self.create_scatter_plot(tsne_base, base_labels)
        novel_scatter = self.create_scatter_plot(tsne_novel, novel_labels, novel=True)
        data = base_scatter + novel_scatter

        plot = go.Figure(data=data)

        return plot

    def create_scatter_plot(self, tsne_results, labels, novel=False):
        data = []
        marker_symbol = "circle-open" if novel else "x"
        legendgroup = "novel" if novel else "base"
        legendgrouptitle_text = "Novel classes" if novel else "Base classes"

        for label in labels.unique():
            indices = torch.where(labels == label)[0]
            if tsne_results.shape[1] == 2:
                scatter = go.Scatter(
                    x=tsne_results[indices, 0],
                    y=tsne_results[indices, 1],
                    mode="markers",
                    marker_symbol=marker_symbol,
                    name=f"Class {label+1}",
                    legendgroup=legendgroup,
                    legendgrouptitle_text=legendgrouptitle_text,
                )
            elif tsne_results.shape[1] == 3:
                scatter = go.Scatter3d(
                    x=tsne_results[indices, 0],
                    y=tsne_results[indices, 1],
                    z=tsne_results[indices, 2],
                    mode="markers",
                    marker_symbol=marker_symbol,
                    name=f"Class {label+1}",
                    legendgroup=legendgroup,
                    legendgrouptitle_text=legendgrouptitle_text,
                )
            else:
                raise NotImplementedError
            data.append(scatter)
        return data


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
