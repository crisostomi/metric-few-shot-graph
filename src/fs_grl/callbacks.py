import json
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

from fs_grl.data.utils import get_label_to_samples_map
from fs_grl.modules.components.node_embedder import NodeEmbedder

pylogger = logging.getLogger(__name__)


class TSNEPlot(Callback):
    def __init__(self, samples_per_class, colors_path, threedimensional) -> None:
        super().__init__()
        self.samples_per_class = samples_per_class
        with open(colors_path) as f:
            colors = json.load(f)
        self.colors = colors
        self.threedimensional = threedimensional

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        split_base_novel_samples = trainer.datamodule.split_base_novel_samples()
        train_dataset, val_dataset, novel_dataset = (
            split_base_novel_samples["base"],
            split_base_novel_samples["val"],
            split_base_novel_samples["novel"],
        )
        train_dataset = self.sample_data_tsne(train_dataset, self.samples_per_class)
        val_dataset = self.sample_data_tsne(val_dataset, self.samples_per_class)
        novel_dataset = self.sample_data_tsne(novel_dataset, self.samples_per_class)

        #

        dataset = train_dataset + val_dataset + novel_dataset

        dataloader = DataLoader(dataset)
        embedder = pl_module.model.embedder
        embedder.eval()

        embeddings = []
        classes = []
        for batch in dataloader:
            batch.to(pl_module.device)

            embedded_sample = embedder(batch)
            if isinstance(embedder, NodeEmbedder):
                aggregator_indices = batch.ptr[1:] - 1
                embedded_sample = embedded_sample[aggregator_indices]

            embeddings.append(embedded_sample.cpu())
            classes.append(batch.y.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        classes = torch.cat(classes, dim=0)
        assert embeddings.size(0) == classes.size(0)

        tsne_results = self.compute_tsne(n_components=2, embeddings=embeddings)
        plot = self.tsne_plot(
            tsne_results=tsne_results,
            labels=classes,
            num_train_samples=len(train_dataset),
            num_val_samples=len(val_dataset),
            num_novel_samples=len(novel_dataset),
        )

        trainer.logger.experiment.log({"t-SNE": plot})

        if self.threedimensional:
            tsne3d_results = self.compute_tsne(n_components=3, embeddings=embeddings)
            plot3d = self.tsne_plot(
                tsne_results=tsne3d_results,
                labels=classes,
                num_train_samples=len(train_dataset),
                num_val_samples=len(val_dataset),
                num_novel_samples=len(novel_dataset),
            )

            trainer.logger.experiment.log({"3d t-SNE": plot3d})

    def sample_data_tsne(self, dataset, num_samples):
        label_to_samples_map = get_label_to_samples_map(dataset)
        sampled_data = []
        for _, samples in label_to_samples_map.items():
            idxs = np.arange(len(samples))
            np.random.shuffle(idxs)
            upper_bound = min(len(samples), num_samples)
            sampled_data += [samples[idx] for idx in idxs[:upper_bound]]
        return sampled_data

    def compute_tsne(self, n_components, embeddings):
        pylogger.info(f"Computing {n_components}d t-SNE...")

        tsne = TSNE(n_components=n_components, n_iter=1000)
        tsne_results = tsne.fit_transform(embeddings)

        return tsne_results

    def tsne_plot(self, tsne_results, labels, num_train_samples, num_val_samples, num_novel_samples):
        tsne_results = torch.from_numpy(tsne_results)
        tsne_train, tsne_val, tsne_novel = tsne_results.split(
            tuple([num_train_samples, num_val_samples, num_novel_samples])
        )

        train_labels, val_labels, novel_labels = labels.split(
            tuple([num_train_samples, num_val_samples, num_novel_samples])
        )

        assert len(tsne_results) == (len(tsne_train) + len(tsne_val) + len(tsne_novel))
        assert len(labels) == (len(train_labels) + len(val_labels) + len(novel_labels))

        train_scatter = self.create_scatter_plot(tsne_train, train_labels, split="train")
        val_scatter = self.create_scatter_plot(tsne_val, val_labels, split="val")
        novel_scatter = self.create_scatter_plot(tsne_novel, novel_labels, split="novel")
        data = train_scatter + val_scatter + novel_scatter

        plot = go.Figure(data=data)

        return plot

    def create_scatter_plot(self, tsne_results, labels, split):
        data = []
        marker_symbol = "circle-open" if split == "novel" else "x"
        legendgroup = split
        legendgrouptitle_text = f"{split} classes"

        for label in labels.unique():
            indices = torch.where(labels == label)[0]
            rgb = self.colors[label.item()]
            if tsne_results.shape[1] == 2:
                scatter = go.Scatter(
                    x=tsne_results[indices, 0],
                    y=tsne_results[indices, 1],
                    mode="markers",
                    marker_symbol=f"{marker_symbol}",
                    marker_color=f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})",
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
                    marker_size=3 if marker_symbol == "x" else 6,
                    marker_color=f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})",
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
