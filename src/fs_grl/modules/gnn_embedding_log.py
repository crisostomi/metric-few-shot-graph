from fs_grl.modules.gnn_embedding_similarity import GNNEmbeddingSimilarity
from fs_grl.modules.losses.margin import MarginLoss
from fs_grl.modules.similarities.cosine import cosine


class GNNEmbeddingLog(GNNEmbeddingSimilarity):
    def __init__(self, cfg, feature_dim, num_classes, margin, **kwargs):
        super().__init__(cfg, feature_dim=feature_dim, num_classes=num_classes, **kwargs)
        self.loss_func = MarginLoss(margin=margin, reduction="mean")

    def get_similarities(self, batch_queries, batch_prototypes):
        """

        :param batch_queries ~ (num_queries_batch*num_classes, hidden_dim)
        :param batch_prototypes ~ (num_queries_batch*num_classes, hidden_dim)
        :return:
        """

        similarities = cosine(batch_queries, batch_prototypes)

        return similarities
