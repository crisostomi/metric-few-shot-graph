import torch
from torch.nn import functional as F

from fs_grl.data.episode import EpisodeHParams
from fs_grl.modules.gnn_embedding_similarity import GNNEmbeddingSimilarity
from fs_grl.modules.losses.margin import MarginLoss


class GNNEmbeddingCosine(GNNEmbeddingSimilarity):
    def __init__(
        self, cfg, feature_dim, hidden_dim, embedding_dim, num_classes, episode_hparams: EpisodeHParams, **kwargs
    ):
        super().__init__(
            cfg,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            episode_hparams=episode_hparams,
        )
        self.loss_func = MarginLoss(margin=0.5, reduction="mean")

    def get_similarities(self, batch_queries, batch_prototypes):
        """

        :param batch_queries ~ (num_queries_batch*num_classes, hidden_dim)
        :param batch_prototypes ~ (num_queries_batch*num_classes, hidden_dim)
        :return:
        """
        similarities = torch.einsum("qh,qh->q", (F.normalize(batch_queries), F.normalize(batch_prototypes)))

        return similarities
