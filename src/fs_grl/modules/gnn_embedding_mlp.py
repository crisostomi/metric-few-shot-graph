import torch
from hydra.utils import instantiate
from torch.nn import functional as F

from fs_grl.modules.gnn_embedding_pairwise import GNNEmbeddingPairwise
from fs_grl.modules.losses.margin import MarginLoss


class GNNEmbeddingMLP(GNNEmbeddingPairwise):
    def __init__(self, cfg, feature_dim, num_classes, margin, **kwargs):
        super().__init__(cfg, feature_dim=feature_dim, num_classes=num_classes, margin=margin)

        self.similarity_network = instantiate(
            self.cfg.similarity_network,
            input_dim=self.embedder.embedding_dim * 2,
            hidden_dim=self.embedder.embedding_dim,
        )

        self.loss_func = MarginLoss(margin=margin, reduction="mean")

    def get_similarities(self, embedded_queries, class_prototypes, batch):

        batch_queries_prototypes = self.align_queries_prototypes(batch, embedded_queries, class_prototypes)
        batch_queries, batch_prototypes = batch_queries_prototypes["queries"], batch_queries_prototypes["prototypes"]

        batch_queries = F.normalize(batch_queries, dim=-1)
        batch_prototypes = F.normalize(batch_prototypes, dim=-1)

        merged_query_prototypes = torch.cat((batch_queries, batch_prototypes), dim=-1)

        similarities = self.similarity_network(merged_query_prototypes)

        similarities = F.tanh(similarities)

        return similarities
