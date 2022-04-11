import abc
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch

from fs_grl.data.episode import EpisodeBatch


class PrototypicalDML(abc.ABC, nn.Module):
    def __init__(self):
        super().__init__()

    def embed_supports(self, batch: EpisodeBatch) -> torch.Tensor:
        """
        :param batch: Batch containing BxNxK support graphs as a single large graph
        :return: embedded supports ~ ((B*N*K)xE), each graph embedded as a point in R^{E}
        """
        return self._embed(batch.supports)

    def embed_queries(self, batch: EpisodeBatch) -> torch.Tensor:
        """
        :param batch: Batch containing BxNxQ query graphs
        :return: embedded queries ~ (BxNxQxE), each graph embedded as a point in R^{E}
        """
        return self._embed(batch.queries)

    def _embed(self, batch: Batch) -> torch.Tensor:
        """
        Embeds a batch of graphs given as a single large graph

        :param batch: Batch containing graphs to embed
        :return: embedded graphs, each graph embedded as a point in R^{E}
        """

        embedded_batch = self.embedder(batch)
        return embedded_batch

    def get_sample_prototypes_similarities(
        self, sample: torch.Tensor, prototypes: torch.Tensor, batch: EpisodeBatch
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_prototypes(self, embedded_supports: torch.Tensor, batch: EpisodeBatch) -> List[Dict[int, torch.Tensor]]:
        """

        :param embedded_supports: (num_supports_in_batch, embedding_dim)
        :param batch:
        :return: a list (num_episodes,) where each entry is a dict { label_1: label_1_prototype, ... }
                containing for each label in the episode the corresponding prototype
        """
        pass

    def align_queries_prototypes(
        self,
        batch: EpisodeBatch,
        embedded_queries: torch.Tensor,
        label_to_embedded_prototypes: List[Dict[int, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Aligns query and prototype embeddings, returning two matrices of shape (B*N*Q*N, embedding_dim)

        :param batch: EpisodeBatch
        :param embedded_queries: shape (num_queries_batch, hidden_dim)
        :param label_to_embedded_prototypes: list (num_episodes) containing for each episode
                                the mapping global label -> corresponding embedded prototype
        :return:
        """

        num_episodes = batch.num_episodes

        embedded_queries_per_episode = embedded_queries.split(tuple([batch.num_queries_per_episode] * num_episodes))

        batch_queries = []
        batch_prototypes = []
        for episode in range(num_episodes):

            class_prototype_matrix = self.get_prototype_matrix_from_dict(label_to_embedded_prototypes[episode])

            aligned_queries, aligned_prototypes = self.align_queries_prototypes_episode(
                embedded_queries_per_episode[episode], class_prototype_matrix, batch
            )

            batch_queries.append(aligned_queries)
            batch_prototypes.append(aligned_prototypes)

        return {"queries": torch.cat(batch_queries, dim=0), "prototypes": torch.cat(batch_prototypes, dim=0)}

    def align_queries_prototypes_episode(self, queries, prototypes_matrix, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Aligns query and prototype embeddings for a single episode,
        returning two matrices of shape (N*Q*N, embedding_dim)

        :param queries: (N*Q*N, hidden_dim)
        :param prototypes_matrix: (N, hidden_dim)
        :param batch:
        :return: aligned embedded queries and prototypes, as matrices of shape (N*Q*N, hidden_dim)
        """
        # shape (N*Q*N, hidden_dim)
        aligned_embedded_queries = queries.repeat_interleave(batch.episode_hparams.num_classes_per_episode, dim=0)

        # shape (N*Q*N, hidden_dim)
        aligned_prototypes = prototypes_matrix.repeat((batch.num_queries_per_episode, 1))

        return aligned_embedded_queries, aligned_prototypes

    @classmethod
    def get_prototype_matrix_from_dict(cls, label_to_prototype_embeddings: Dict) -> torch.Tensor:
        """
        Returns a matrix where row i contains the embedded class prototype
        for the i-th label in the sorted array of global labels

        :param label_to_prototype_embeddings: mapping global label -> corresponding embedding
        :return class_prototype_matrix: tensor (num_classes_episode, embedding_dim)
        """

        label_and_embedded_prototypes_tuples = [
            (global_class, prototype) for global_class, prototype in label_to_prototype_embeddings.items()
        ]
        label_and_embedded_prototypes_tuples.sort(key=lambda tup: tup[0])
        sorted_class_prototypes_tensors = [tup[1] for tup in label_and_embedded_prototypes_tuples]

        # shape (num_classes_episode, embedding_dim)
        class_prototype_matrix = torch.stack(sorted_class_prototypes_tensors)

        return class_prototype_matrix

    def compute_sample_prototypes_similarities(
        self, sample: torch.Tensor, label_to_prototype_embeddings: Dict, batch: EpisodeBatch
    ) -> torch.Tensor:
        """
        Computes the similarities between given sample and label prototypes

        :param sample:
        :param label_to_prototype_embeddings: mapping global label -> corresponding prototype embedding
        :param batch:

        :return: similarities, shape (num_classes_per_episode)
        """
        class_prototype_matrix = self.get_prototype_matrix_from_dict(label_to_prototype_embeddings)

        # shape (num_classes_per_episode)
        similarities = self.get_sample_prototypes_similarities(sample, class_prototype_matrix, batch)

        return similarities

    def compute_crossover_regularizer(self, model_out: Dict, batch: EpisodeBatch):
        """
        Computes the regularizer term for the artificial samples created as cross-over of samples
        from different classes

        :param model_out:
        :param batch:
        :return:
        """

        query_embeddings = model_out["embedded_queries"]

        global_labels_by_episode = batch.get_global_labels_by_episode()
        query_labels_by_episode = batch.get_query_labels_by_episode()

        query_embeddings_by_episode = query_embeddings.split(
            tuple([batch.num_queries_per_episode] * batch.num_episodes)
        )

        regularizer_term = 0
        for episode in range(batch.num_episodes):
            episode_prototypes = model_out["class_prototypes"][episode]
            episode_global_label_pairs = self.get_global_label_pairs(global_labels_by_episode[episode])
            episode_query_embeddings = query_embeddings_by_episode[episode]
            episode_query_labels = query_labels_by_episode[episode]

            episode_regularizer_term = self.compute_episode_crossover_regularizer(
                episode_prototypes, episode_global_label_pairs, episode_query_embeddings, episode_query_labels, batch
            )
            regularizer_term += episode_regularizer_term

        return regularizer_term / batch.num_episodes

    def compute_episode_crossover_regularizer(
        self, episode_prototypes, episode_global_label_pairs, episode_query_embeddings, episode_query_labels, batch
    ):
        episode_regularizer_term = 0

        for pair in episode_global_label_pairs:
            label_a, label_b = pair
            label_a_query = self.sample_query_embedding(label_a, episode_query_embeddings, episode_query_labels, batch)
            label_b_query = self.sample_query_embedding(label_b, episode_query_embeddings, episode_query_labels, batch)

            alpha = torch.rand(1).type_as(label_a_query)

            crossover = self.create_crossover(label_a_query, label_b_query, alpha)

            crossover_class_distr = self.get_sample_class_distribution(crossover, episode_prototypes, batch)
            label_a_class_distr = self.get_sample_class_distribution(label_a_query, episode_prototypes, batch)
            label_b_class_distr = self.get_sample_class_distribution(label_b_query, episode_prototypes, batch)

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

        :return:
        """

        gating_vector = self.construct_macro_features_gating_vector(alpha)

        crossover = gating_vector * sample_a + (1 - gating_vector) * sample_b

        return crossover

    def get_global_label_pairs(self, episode_global_labels):
        """

        :param episode_global_labels:
        :return:
        """

        episode_pairs = []
        for ind_a, label_a in enumerate(episode_global_labels):

            for ind_b, label_b in enumerate(episode_global_labels[ind_a + 1 :]):
                episode_pairs.append((label_a, label_b))

        return episode_pairs

    def construct_gating_vector(self, alpha: torch.Tensor):
        """

        :param alpha: mixing ratio
        :return:
        """
        emb_dim = self.embedder.embedding_dim

        num_features_to_sample = int(alpha * emb_dim)
        random_indices = np.random.choice(np.arange(0, emb_dim), size=num_features_to_sample)

        gating_vector = torch.zeros((emb_dim,)).type_as(alpha)

        gating_vector[random_indices] = 1

        return gating_vector

    def construct_macro_features_gating_vector(self, alpha):
        """
        :param alpha: mixing ratio

        :return:
        """
        emb_dim = self.embedder.embedding_dim

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

    def get_sample_class_distribution(
        self, sample: torch.Tensor, label_to_prototype_embeddings: Dict, batch: EpisodeBatch
    ):
        """
        :param sample:
        :param label_to_prototype_embeddings:
        :param batch:
        :return:
        """
        sampler_to_prototypes_similarities = self.compute_sample_prototypes_similarities(
            sample, label_to_prototype_embeddings, batch
        )
        sample_class_distr = torch.softmax(sampler_to_prototypes_similarities, dim=-1)

        return sample_class_distr

    def sample_query_embedding(
        self, label: torch.Tensor, query_embeddings: torch.Tensor, query_labels: torch.Tensor, batch: EpisodeBatch
    ):
        """
        Sample one query embedding for the given label
        :return:
        """
        assert query_embeddings.shape[0] == query_labels.shape[0]

        indices = torch.where(query_labels == label.item())[0]
        assert len(indices) == batch.episode_hparams.num_queries_per_class

        # just take the first embedding having label "label"
        query_embedding = query_embeddings[indices[0]]

        return query_embedding

    def get_intra_class_variance(
        self, embedded_supports: torch.Tensor, label_to_prototype_embed_map: Dict, batch: EpisodeBatch
    ):
        """
        Computes the mean of the intra-class variance for each episode, which is the
        sum of the squared l2 distance between embedded supports and their corresponding prototypes

        :param embedded_supports:
        :param label_to_prototype_embed_map:
        :param batch:
        :return:
        """

        embedded_supports_by_episode = embedded_supports.split([batch.num_supports_per_episode] * batch.num_episodes)
        labels_by_episode = batch.get_support_labels_by_episode()
        global_labels_by_episode = batch.get_global_labels_by_episode()

        intra_class_var = 0
        for episode_ind in range(batch.num_episodes):

            episode_global_labels = global_labels_by_episode[episode_ind]
            episode_labels = labels_by_episode[episode_ind]
            episode_label_to_prot_embed_map = label_to_prototype_embed_map[episode_ind]
            episode_embedded_supports = embedded_supports_by_episode[episode_ind]

            global_to_local_mapping = {
                global_label.item(): ind for ind, global_label in enumerate(episode_global_labels)
            }

            # map global labels to local labels for the support samples
            episode_local_labels = torch.tensor([global_to_local_mapping[label.item()] for label in episode_labels])

            # (N, embedding_dim)
            class_prototype_matrix = self.get_prototype_matrix_from_dict(episode_label_to_prot_embed_map)

            # (N*K, embedding_dim)
            aligned_class_prototypes = class_prototype_matrix[episode_local_labels]
            distance_from_prototype = torch.pow(
                torch.norm(episode_embedded_supports - aligned_class_prototypes, p=2, dim=-1), 2
            )
            intra_class_var += distance_from_prototype.mean(dim=-1)

        intra_class_var = intra_class_var / batch.num_episodes

        return intra_class_var
