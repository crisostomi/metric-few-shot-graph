from fs_grl.modules.gnn_embedding_pairwise import GNNEmbeddingPairwise
from fs_grl.modules.losses.margin import MarginLoss
from fs_grl.modules.similarities.cosine import cosine


class GNNEmbeddingCosine(GNNEmbeddingPairwise):
    def __init__(self, cfg, feature_dim, num_classes, margin, **kwargs):
        super().__init__(cfg, feature_dim=feature_dim, num_classes=num_classes, **kwargs)
        self.loss_func = MarginLoss(margin=margin, reduction="mean")

    def get_similarities(self, embedded_queries, class_prototypes, batch):
        """

        :param embedded_queries ~
        :param class_prototypes ~
        :return:
        """
        batch_queries_prototypes = self.align_queries_prototypes(batch, embedded_queries, class_prototypes)
        batch_queries, batch_prototypes = batch_queries_prototypes["queries"], batch_queries_prototypes["prototypes"]

        similarities = cosine(batch_queries, batch_prototypes)

        return similarities
