from typing import Dict

import numpy as np
import torch

from fs_grl.data.episode.episode_batch import EpisodeBatch


class MixUpAugmentor:
    def __init__(self, model):
        super().__init__()
        self.model = model

    def compute_latent_mixup_reg(self, model_out: Dict, batch: EpisodeBatch):
        """
        Computes the regularizer term for the artificial samples created as cross-over of samples
        from different classes

        :param model_out:
        :param batch:
        :return
        """

        query_embeddings = model_out["embedded_queries"]

        global_labels_by_episode = batch.get_global_labels_by_episode()
        query_labels_by_episode = batch.get_query_labels_by_episode()

        query_embeddings_by_episode = query_embeddings.split(
            tuple([batch.episode_hparams.num_queries_per_episode] * batch.num_episodes)
        )

        regularizer_term = 0
        for episode in range(batch.num_episodes):
            episode_prototypes = model_out["prototypes_dicts"][episode]
            episode_global_label_pairs = self.get_global_label_pairs(global_labels_by_episode[episode])
            episode_query_embeddings = query_embeddings_by_episode[episode]
            episode_query_labels = query_labels_by_episode[episode]

            episode_regularizer_term = self.compute_episode_crossover_regularizer(
                episode_prototypes,
                episode_global_label_pairs,
                episode_query_embeddings,
                episode_query_labels,
                batch,
            )
            regularizer_term += episode_regularizer_term

        return regularizer_term / batch.num_episodes

    def compute_episode_crossover_regularizer(
        self,
        episode_prototypes,
        episode_global_label_pairs,
        episode_query_embeddings,
        episode_query_labels,
        batch,
    ):
        episode_regularizer_term = 0

        for pair in episode_global_label_pairs:
            label_a, label_b = pair
            label_a_query = self.sample_query_embedding(label_a, episode_query_embeddings, episode_query_labels, batch)
            label_b_query = self.sample_query_embedding(label_b, episode_query_embeddings, episode_query_labels, batch)

            alpha = torch.rand(1).type_as(label_a_query)

            crossover = self.create_crossover(label_a_query, label_b_query, alpha)

            crossover_class_distr = self.model.get_sample_class_distribution(crossover, episode_prototypes, batch)
            label_a_class_distr = self.model.get_sample_class_distribution(label_a_query, episode_prototypes, batch)
            label_b_class_distr = self.model.get_sample_class_distribution(label_b_query, episode_prototypes, batch)

            ground_truth_combination = alpha * label_a_class_distr + (1 - alpha) * label_b_class_distr

            pair_regularizer_term = torch.norm(crossover_class_distr - ground_truth_combination, p=2) ** 2
            episode_regularizer_term += pair_regularizer_term

        return episode_regularizer_term

    def create_crossover(self, sample_a: torch.Tensor, sample_b: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Creates a new sample which is the crossover of sample_a and sample_b

        :param sample_a:
        :param sample_b:
        :param alpha:

        :return
        """

        gating_vector = self.construct_macro_features_gating_vector(alpha)

        crossover = gating_vector * sample_a + (1 - gating_vector) * sample_b

        return crossover

    def get_global_label_pairs(self, episode_global_labels):
        """

        :param episode_global_labels:
        :return
        """

        episode_pairs = []
        for ind_a, label_a in enumerate(episode_global_labels):

            for ind_b, label_b in enumerate(episode_global_labels[ind_a + 1 :]):
                episode_pairs.append((label_a, label_b))

        return episode_pairs

    def construct_gating_vector(self, alpha: torch.Tensor):
        """

        :param alpha: mixing ratio
        :return
        """
        emb_dim = self.model.embedding_dim

        num_features_to_sample = int(alpha * emb_dim)
        random_indices = np.random.choice(np.arange(0, emb_dim), size=num_features_to_sample)

        gating_vector = torch.zeros((emb_dim,)).type_as(alpha)

        gating_vector[random_indices] = 1

        return gating_vector

    def construct_macro_features_gating_vector(self, alpha):
        """
        :param alpha: mixing ratio

        :return
        """
        emb_dim = self.model.embedding_dim

        num_macro_features = np.random.choice(np.arange(10, emb_dim))
        macro_feature_indices = np.arange(0, num_macro_features)

        macro_feature_size = emb_dim // num_macro_features
        num_macro_features_to_sample = int(alpha * num_macro_features)
        random_bin_indices = (
            np.random.choice(macro_feature_indices, size=num_macro_features_to_sample) * macro_feature_size
        )
        indices_repeated = np.repeat(random_bin_indices, axis=0, repeats=macro_feature_size)
        addend = np.tile(np.arange(macro_feature_size), num_macro_features_to_sample)
        feature_to_sample_indices = indices_repeated + addend

        gating_vector = torch.zeros((emb_dim,)).type_as(alpha)

        gating_vector[feature_to_sample_indices] = 1

        return gating_vector

    def sample_query_embedding(
        self, label: torch.Tensor, query_embeddings: torch.Tensor, query_labels: torch.Tensor, batch: EpisodeBatch
    ):
        """
        Sample one query embedding for the given label
        :return
        """
        assert query_embeddings.shape[0] == query_labels.shape[0]

        indices = torch.where(query_labels == label.item())[0]
        assert len(indices) == batch.episode_hparams.num_queries_per_class

        # just take the first embedding having label "label"
        query_embedding = query_embeddings[indices[0]]

        return query_embedding
