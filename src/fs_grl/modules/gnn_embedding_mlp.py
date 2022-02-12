import torch
from torch import nn
from torch.nn import functional as F

from fs_grl.data.episode import EpisodeHParams
from fs_grl.modules.gnn_embedding_similarity import GNNEmbeddingSimilarity
from fs_grl.modules.losses.margin import MarginLoss
from fs_grl.modules.mlp import MLP


class GNNEmbeddingMLP(GNNEmbeddingSimilarity):
    def __init__(
        self,
        cfg,
        feature_dim,
        hidden_dim,
        embedding_dim,
        num_classes,
        episode_hparams: EpisodeHParams,
        num_mlp_layers,
        mlp_non_linearity,
        **kwargs
    ):
        super().__init__(cfg, feature_dim, hidden_dim, embedding_dim, num_classes, episode_hparams)
        self.num_layers = num_mlp_layers
        self.mlp_non_linearity = mlp_non_linearity

        self.similarity_network = MLP(
            num_layers=num_mlp_layers,
            input_dim=embedding_dim * 2,
            output_dim=1,
            hidden_dim=embedding_dim * 2,
            non_linearity=self.mlp_non_linearity,
        )

        self.loss_func = MarginLoss(margin=0.5, reduction="mean")

    def get_similarities(self, batch_queries, batch_prototypes):

        batch_queries = F.normalize(batch_queries, dim=-1)
        batch_prototypes = F.normalize(batch_prototypes, dim=-1)

        merged_query_prototypes = torch.cat((batch_queries, batch_prototypes), dim=-1)

        similarities = self.similarity_network(merged_query_prototypes)

        similarities = nn.functional.tanh(similarities)

        return similarities
