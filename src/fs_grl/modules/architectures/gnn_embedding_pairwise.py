import abc

from fs_grl.data.episode.episode_batch import EpisodeBatch
from fs_grl.modules.architectures.gnn_embedding_similarity import GNNEmbeddingSimilarity


class GNNEmbeddingPairwise(GNNEmbeddingSimilarity, abc.ABC):
    def __init__(self, cfg, feature_dim, num_classes, **kwargs):
        super().__init__(cfg, feature_dim=feature_dim, num_classes=num_classes, **kwargs)

    def compute_classification_loss(self, model_out, batch: EpisodeBatch, **kwargs):
        similarities = model_out["similarities"]

        return self.loss_func(similarities, batch.cosine_targets)
