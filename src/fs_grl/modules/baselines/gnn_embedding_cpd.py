from typing import Dict, List

import torch
from torchcpd.deformable_registration import DeformableRegistration

from fs_grl.data.episode import EpisodeBatch
from fs_grl.modules.baselines.gnn_embedding_pairwise import GNNEmbeddingPairwise
from fs_grl.modules.losses.margin import MarginLoss
from fs_grl.modules.similarities.cosine import cosine


class GNNEmbeddingCPD(GNNEmbeddingPairwise):
    def __init__(self, cfg, feature_dim, num_classes, margin, **kwargs):
        super().__init__(cfg, feature_dim=feature_dim, num_classes=num_classes, **kwargs)
        self.loss_func = MarginLoss(margin=margin, reduction="mean")

    def get_queries_prototypes_similarities_batch(
        self, queries: torch.Tensor, label_to_embedded_prototypes: List[Dict], batch: EpisodeBatch
    ):
        """

        :param queries ~
        :param label_to_embedded_prototypes ~
        :param batch:

        :return:
        """
        batch_queries_prototypes = self.align_queries_prototypes(batch, queries, label_to_embedded_prototypes)
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


if __name__ == "__main__":
    source = torch.randn(50, 3)
    target = torch.randn(90, 3)

    cpd = DeformableRegistration(X=target, Y=source)
    cpd.register()

    cpd_inv = DeformableRegistration(X=source, Y=target)
    cpd_inv.register()
    distance = torch.linalg.norm(cpd.G @ cpd.W).item() + torch.linalg.norm(cpd_inv.G @ cpd_inv.W).item()

    print(distance)
