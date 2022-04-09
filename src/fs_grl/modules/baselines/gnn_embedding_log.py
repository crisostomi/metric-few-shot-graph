from fs_grl.modules.baselines.gnn_embedding_pairwise import GNNEmbeddingPairwise
from fs_grl.modules.losses.log_loss import LogisticLoss
from fs_grl.modules.similarities.cosine import cosine


class GNNEmbeddingLog(GNNEmbeddingPairwise):
    def __init__(self, cfg, feature_dim, num_classes, margin, **kwargs):
        super().__init__(cfg, feature_dim=feature_dim, num_classes=num_classes, **kwargs)
        self.loss_func = LogisticLoss(margin=margin, reduction="mean")

    def get_queries_prototypes_similarities_batch(self, embedded_queries, class_prototypes, batch):
        """

        :param embedded_queries ~
        :param class_prototypes ~
        :return:
        """
        batch_queries_prototypes = self.align_queries_prototypes(batch, embedded_queries, class_prototypes)
        batch_queries, batch_prototypes = batch_queries_prototypes["queries"], batch_queries_prototypes["prototypes"]

        similarities = cosine(batch_queries, batch_prototypes)

        return similarities
