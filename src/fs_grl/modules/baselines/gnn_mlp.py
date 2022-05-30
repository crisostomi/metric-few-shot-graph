import logging

import torch.nn as nn
from hydra.utils import instantiate
from torch_geometric.data import Batch

from fs_grl.modules.graph_embedder import GraphEmbedder

pylogger = logging.getLogger(__name__)


class GNN_MLP(nn.Module):
    def __init__(self, cfg, feature_dim, num_classes, **kwargs):
        super().__init__()

        self.embedder: GraphEmbedder = instantiate(
            cfg.embedder, cfg=cfg.embedder, feature_dim=feature_dim, _recursive_=False
        )

        self.classifier = instantiate(cfg.classifier, output_dim=num_classes)

    def forward(self, batch: Batch):

        embeddings = self.embedder(batch)

        logits = self.classifier(embeddings)

        return logits
