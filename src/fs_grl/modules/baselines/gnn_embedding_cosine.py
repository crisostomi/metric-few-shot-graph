import torch

from fs_grl.data.episode import EpisodeBatch
from fs_grl.modules.baselines.gnn_embedding_pairwise import GNNEmbeddingPairwise
from fs_grl.modules.losses.margin import MarginLoss
from fs_grl.modules.similarities.cosine import cosine


class GNNEmbeddingCosine(GNNEmbeddingPairwise):
    def __init__(self, cfg, feature_dim, num_classes, margin, **kwargs):
        super().__init__(cfg, feature_dim=feature_dim, num_classes=num_classes, **kwargs)
        self.loss_func = MarginLoss(margin=margin, reduction="mean")

    def get_queries_prototypes_similarities_batch(
        self, queries: torch.Tensor, prototypes: torch.Tensor, batch: EpisodeBatch
    ):
        """

        :param queries ~
        :param prototypes ~
        :return:
        """
        batch_queries_prototypes = self.align_queries_prototypes(batch, queries, prototypes)
        batch_queries, batch_prototypes = batch_queries_prototypes["queries"], batch_queries_prototypes["prototypes"]

        similarities = cosine(batch_queries, batch_prototypes)

        return similarities

    def get_sample_prototypes_similarities(
        self, sample: torch.Tensor, prototypes: torch.Tensor, batch: EpisodeBatch
    ) -> torch.Tensor:
        """
        :param sample:
        :param prototypes:
        :param batch:
        :return:
        """
        N = batch.episode_hparams.num_classes_per_episode
        repeated_sample = sample.repeat((N, 1))

        return cosine(repeated_sample, prototypes)
