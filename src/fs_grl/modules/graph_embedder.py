import logging

import torch.nn as nn
from hydra.utils import instantiate
from torch_geometric.nn import GraphMultisetTransformer, JumpingKnowledge

pylogger = logging.getLogger(__name__)


class GraphEmbedder(nn.Module):
    def __init__(self, cfg, feature_dim, **kwargs):
        super().__init__()

        self.feature_dim = feature_dim

        self.node_embedder = instantiate(cfg.node_embedder, feature_dim=self.feature_dim, _recursive_=True)

        self.pooling = instantiate(cfg.pooling) if cfg.pooling is not None else None

        self.jumping_knowledge = JumpingKnowledge(mode="cat")  # TODO: maybe to delete

    @property
    def embedding_dim(self):
        return self.node_embedder.embedding_dim

    def forward(self, batch):
        """
        Embeds a batch of graphs given as a single large graph

        :param batch: Batch containing graphs to embed
        :return: embedded graphs, each graph embedded as a point in R^{E}
        """

        node_embeddings = self.node_embedder(batch)

        # pooled_out ~ (num_samples_in_batch, embedding_dim)
        if self.pooling is not None:
            pooling_args = self.get_pooling_args(node_embeddings, batch)
            pooled_out = self.pooling(**pooling_args)
            return pooled_out

        return node_embeddings

    def get_pooling_args(self, node_out_features, batch):
        pooling_args = {"x": node_out_features, "batch": batch.batch}
        if isinstance(self.pooling, GraphMultisetTransformer):
            pooling_args["edge_index"] = batch.edge_index
        return pooling_args
