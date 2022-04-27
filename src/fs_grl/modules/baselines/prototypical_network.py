from typing import Dict

import torch
from torch.nn import NLLLoss

from fs_grl.data.episode import EpisodeBatch
from fs_grl.modules.baselines.gnn_embedding_similarity import GNNEmbeddingSimilarity

# from fs_grl.modules.similarities.cosine import cosine
from fs_grl.modules.similarities.squared_l2 import squared_l2


class PrototypicalNetwork(GNNEmbeddingSimilarity):
    def __init__(self, cfg, feature_dim, num_classes, **kwargs):
        super().__init__(cfg, feature_dim=feature_dim, num_classes=num_classes, **kwargs)
        self.loss_func = NLLLoss()
        self.register_buffer("metric_scaling_factor", torch.tensor(7.5))

    def forward(self, batch: EpisodeBatch):
        """
        :param batch:
        :return:
        """

        graph_level = not self.prototypes_from_nodes
        embedded_supports = self.embed_supports(batch, graph_level=graph_level)

        # shape (num_queries_batch, hidden_dim)
        embedded_queries = self.embed_queries(batch)

        # shape (num_classes_per_episode, hidden_dim)
        class_prototypes = self.get_prototypes(embedded_supports, batch)

        distances = self.get_queries_prototypes_correlations_batch(embedded_queries, class_prototypes, batch)
        distances = self.metric_scaling_factor * distances

        return {
            "embedded_queries": embedded_queries,
            "class_prototypes": class_prototypes,
            "distances": distances,
        }

    def get_queries_prototypes_correlations_batch(
        self, embedded_queries: torch.Tensor, class_prototypes: torch.Tensor, batch: EpisodeBatch
    ):
        """

        :param embedded_queries ~ (num_queries_batch*num_classes, hidden_dim)
        :param class_prototypes ~ (num_queries_batch*num_classes, hidden_dim)
        :param batch:

        :return:
        """
        batch_queries_prototypes = self.align_queries_prototypes(batch, embedded_queries, class_prototypes)
        batch_queries, batch_prototypes = batch_queries_prototypes["queries"], batch_queries_prototypes["prototypes"]

        distances = squared_l2(batch_queries, batch_prototypes)

        return distances

    def compute_loss(self, model_out, batch, **kwargs):
        # shape (B, N*Q, N)
        distances = model_out["distances"]

        distances = distances.view(batch.num_episodes, -1, batch.episode_hparams.num_classes_per_episode)
        probabilities = torch.log_softmax(-distances, dim=-1)

        labels_per_episode = batch.local_labels.view(batch.num_episodes, -1)

        cum_loss = 0
        for episode in range(batch.num_episodes):
            cum_loss += self.loss_func(probabilities[episode], labels_per_episode[episode])

        cum_loss /= batch.num_episodes
        return cum_loss

    def get_sample_prototypes_correlations(
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

        return squared_l2(repeated_sample, prototypes)

    def get_predictions(self, step_out: Dict, batch: EpisodeBatch) -> torch.Tensor:
        """

        :param similarities: shape (B * N*Q * N)
        :param batch:

        :return:
        """
        # shape ~(num_episodes * num_queries_per_class * num_classes_per_episode)
        distances = step_out["model_out"]["distances"]

        distances = distances.view(batch.num_episodes, -1, batch.episode_hparams.num_classes_per_episode)
        probabilities = torch.log_softmax(-distances, dim=-1)

        # shape (B*(N*Q)) contains for each query the most similar label
        pred_labels = torch.argmax(probabilities, dim=-1)

        pred_global_labels = self.map_pred_labels_to_global(
            pred_labels=pred_labels, batch_global_labels=batch.global_labels, num_episodes=batch.num_episodes
        )

        return pred_global_labels

    def get_sample_class_distribution(
        self, sample: torch.Tensor, label_to_prototype_embeddings: Dict, batch: EpisodeBatch
    ):
        """
        :param sample:
        :param label_to_prototype_embeddings:
        :param batch:
        :return:
        """
        sampler_to_prototypes_similarities = self.compute_sample_prototypes_correlations(
            sample, label_to_prototype_embeddings, batch
        )
        sample_class_distr = torch.softmax(-sampler_to_prototypes_similarities, dim=-1)

        return sample_class_distr
