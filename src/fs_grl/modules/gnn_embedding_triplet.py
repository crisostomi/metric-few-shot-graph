from typing import Dict, List

import torch
from torch import nn

from fs_grl.data.episode import EpisodeBatch
from fs_grl.modules.gnn_embedding_similarity import GNNEmbeddingSimilarity
from fs_grl.modules.similarities.cosine import cosine, cosine_distance_1D


class GNNEmbeddingTriplet(GNNEmbeddingSimilarity):
    def __init__(self, cfg, feature_dim, num_classes, margin, **kwargs):
        super().__init__(cfg, feature_dim=feature_dim, num_classes=num_classes, **kwargs)
        self.loss_func = nn.TripletMarginWithDistanceLoss(
            margin=margin, reduction="mean", distance_function=cosine_distance_1D
        )

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

    def compute_loss(self, model_out, batch: EpisodeBatch, **kwargs):

        # ( B*(N*Q), D)
        embedded_queries = model_out["embedded_queries"]
        # list of len (B) of dicts { glob_label: prototype }, where each dict has
        # the global labels of the episode as keys
        class_prototypes = model_out["class_prototypes"]

        aligned_embeddings = self.align_triplets(queries=embedded_queries, prototypes=class_prototypes, batch=batch)

        # shape (B*(N*Q)*(N-1), D)
        queries, positives, negatives = (
            aligned_embeddings["queries"],
            aligned_embeddings["positives"],
            aligned_embeddings["negatives"],
        )

        return self.loss_func(queries, positives, negatives)

    def align_triplets(self, batch, queries: torch.Tensor, prototypes: List[Dict[int, torch.Tensor]]):
        """

        :param batch:
        :param queries: shape (num_queries_batch, hidden_dim)
        :param prototypes:
        :return:
        """

        num_queries_per_episode = (
            batch.episode_hparams.num_queries_per_class * batch.episode_hparams.num_classes_per_episode
        )

        embedded_queries_per_episode = queries.split(tuple([num_queries_per_episode] * batch.num_episodes))
        local_labels_per_episode = batch.local_labels.split(tuple([num_queries_per_episode] * batch.num_episodes))

        batch_queries = []
        batch_positives = []
        batch_negatives = []

        for episode in range(batch.num_episodes):
            # shape (N, D)
            class_prototype_matrix = self.get_prototype_matrix_from_dict(prototypes[episode])

            queries, positives, negatives = self.align_queries_positives_negatives_episode(
                embedded_queries_per_episode[episode], class_prototype_matrix, local_labels_per_episode[episode], batch
            )

            batch_queries.append(queries)
            batch_positives.append(positives)
            batch_negatives.append(negatives)

        return {
            "queries": torch.cat(batch_queries, dim=0),
            "positives": torch.cat(batch_positives, dim=0),
            "negatives": torch.cat(batch_negatives, dim=0),
        }

    def align_queries_positives_negatives_episode(self, queries, prototype_matrix, labels, batch):
        """
        :param queries: (num_queries_episode, hidden_dim)
        :param prototypes:
        :return:
        """
        N = batch.episode_hparams.num_classes_per_episode

        # ( (N*Q)*(N-1), D)
        queries_aligned = queries.repeat_interleave(N - 1, dim=0)
        embedding_dim = queries.shape[-1]

        positives = prototype_matrix[labels]
        positives_aligned = positives.repeat_interleave(N - 1, dim=0)

        not_Y = torch.tensor([[i for i in range(N) if i != y] for y in labels])
        negatives = prototype_matrix[not_Y]
        negatives_aligned = negatives.reshape(-1, embedding_dim)

        return queries_aligned, positives_aligned, negatives_aligned
