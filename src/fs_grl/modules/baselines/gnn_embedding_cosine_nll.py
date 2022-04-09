import torch
from torch.nn import NLLLoss

from fs_grl.data.episode import EpisodeBatch
from fs_grl.modules.baselines.gnn_embedding_similarity import GNNEmbeddingSimilarity
from fs_grl.modules.similarities.cosine import cosine


class GNNEmbeddingCosineNLL(GNNEmbeddingSimilarity):
    def __init__(self, cfg, feature_dim, num_classes, **kwargs):
        super().__init__(cfg, feature_dim=feature_dim, num_classes=num_classes, **kwargs)
        self.loss_func = NLLLoss()

    def get_queries_prototypes_similarities_batch(
        self, embedded_queries: torch.Tensor, class_prototypes: torch.Tensor, batch: EpisodeBatch
    ):
        """

        :param embedded_queries ~ (num_queries_batch*num_classes, hidden_dim)
        :param class_prototypes ~ (num_queries_batch*num_classes, hidden_dim)

        :return:
        """
        batch_queries_prototypes = self.align_queries_prototypes(batch, embedded_queries, class_prototypes)
        batch_queries, batch_prototypes = batch_queries_prototypes["queries"], batch_queries_prototypes["prototypes"]

        similarities = cosine(batch_queries, batch_prototypes)

        return similarities

    def compute_loss(self, model_out, batch, **kwargs):
        # shape (B, N*Q, N)
        similarities = model_out["similarities"]

        similarities = similarities.reshape(batch.num_episodes, -1, batch.episode_hparams.num_classes_per_episode)
        probabilities = torch.log_softmax(similarities, dim=-1)

        labels_per_episode = batch.local_labels.reshape(batch.num_episodes, -1)

        cum_loss = 0
        for episode in range(batch.num_episodes):
            cum_loss += self.loss_func(probabilities[episode], labels_per_episode[episode])

        cum_loss /= batch.num_episodes
        return cum_loss

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
