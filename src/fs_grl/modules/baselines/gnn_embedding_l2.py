from fs_grl.modules.baselines.gnn_embedding_pairwise import GNNEmbeddingPairwise
from fs_grl.modules.losses.margin import MarginLoss


class GNNEmbeddingL2(GNNEmbeddingPairwise):
    def __init__(self, cfg, feature_dim, num_classes, margin, **kwargs):
        super().__init__(cfg, feature_dim=feature_dim, num_classes=num_classes, **kwargs)
        self.loss_func = MarginLoss(margin=margin, reduction="mean")

    # def get_queries_prototypes_correlations_batch(self, embedded_queries, class_prototypes, batch):
    #     """
    #
    #     :param batch_queries ~ (num_queries_batch*num_classes, hidden_dim)
    #     :param batch_prototypes ~ (num_queries_batch*num_classes, hidden_dim)
    #     :return:
    #     """
    #     batch_queries_prototypes = self.align_queries_prototypes(batch, embedded_queries, class_prototypes)
    #     batch_queries, batch_prototypes = batch_queries_prototypes["queries"], batch_queries_prototypes["prototypes"]
    #
    #     return similarities
