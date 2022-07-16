import abc
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch_geometric.data import Batch

from fs_grl.data.episode.episode_batch import EpisodeBatch
from fs_grl.modules.components.mixup_augmentor import MixUpAugmentor, MolecularMixUpAugmentor


class PrototypeBased(abc.ABC, nn.Module):
    """
    Abstract class for DML architectures that involve some sort of prototypes.
    """

    def __init__(self, loss_weights):
        super().__init__()
        self.loss_weights = loss_weights
        self.mixup_augmentor = MixUpAugmentor(self)

    def embed_supports(self, batch: EpisodeBatch) -> torch.Tensor:
        """
        :param batch: Batch containing BxNxK support graphs as a single large graph
        :return embedded supports ~ ((B*N*K), E), each graph embedded as a point in R^E
        """
        return self._embed(batch.supports)

    def embed_queries(self, batch: EpisodeBatch) -> torch.Tensor:
        """
        :param batch: Batch containing BxNxQ query graphs
        :return embedded queries ~ (BxNxQxE), each graph embedded as a point in R^{E}
        """
        return self._embed(batch.queries)

    def _embed(self, batch: Batch) -> torch.Tensor:
        """
        Embeds a batch of graphs given as a single large graph

        :param batch: Batch containing graphs to embed
        :return embedded graphs, each graph embedded as a point in R^{E}
        """

        embedded_batch = self.embedder(batch)
        return embedded_batch

    def compute_sample_prototypes_correlations(
        self, sample: torch.Tensor, prototypes: torch.Tensor, batch: EpisodeBatch
    ) -> torch.Tensor:
        f"""
        Get similarity or distance between {sample} and {prototypes}

        :param sample:
        :param prototypes:
        :param batch:

        :return:
        """
        pass

    @abc.abstractmethod
    def compute_prototypes(
        self, embedded_supports: torch.Tensor, batch: EpisodeBatch, **kwargs
    ) -> List[Dict[int, torch.Tensor]]:
        """
        Compute the class prototypes given the embedded supports.

        :param embedded_supports: (num_supports_in_batch, embedding_dim)
        :param batch:
        :return a list (num_episodes,) where each entry is a dict { label_1: label_1_prototype, ... }
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
        Aligns query and prototype embeddings to compute similarities,
        returning two matrices of shape (B*N*Q*N, embedding_dim), as for each episode
        each of the N*Q queries must be compared with N prototypes.

        :param batch: EpisodeBatch
        :param embedded_queries: shape (num_queries_batch, hidden_dim)
        :param label_to_embedded_prototypes: list (num_episodes) containing for each episode
                                the mapping global label -> corresponding embedded prototype
        :return
        """

        num_episodes = batch.num_episodes

        embedded_queries_per_episode = embedded_queries.view(
            num_episodes, batch.episode_hparams.num_queries_per_episode, self.embedder.embedding_dim
        )

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
        :return aligned embedded queries and prototypes, as matrices of shape (N*Q*N, hidden_dim)
        """
        # shape (N*Q*N, hidden_dim)
        aligned_embedded_queries = queries.repeat_interleave(batch.episode_hparams.num_classes_per_episode, dim=0)

        # shape (N*Q*N, hidden_dim)
        aligned_prototypes = prototypes_matrix.repeat((batch.episode_hparams.num_queries_per_episode, 1))

        return aligned_embedded_queries, aligned_prototypes

    def get_sample_prototypes_correlations(
        self, sample: torch.Tensor, label_to_prototype_embeddings: Dict, batch: EpisodeBatch
    ) -> torch.Tensor:
        f"""
        Obtains the similarities between {sample} and prototypes. This is done by first obtaining a
        prototype matrix from the label -> prototype mapping and then computing the correlations from
        the sample vector and the matrix.

        :param sample: tensor ~ (embedding_dim)
        :param label_to_prototype_embeddings: mapping global label -> corresponding prototype embedding
        :param batch:

        :return similarities: tensor ~ (num_classes_per_episode)
        """
        class_prototype_matrix = self.get_prototype_matrix_from_dict(label_to_prototype_embeddings)

        # shape (num_classes_per_episode)
        correlations = self.compute_sample_prototypes_correlations(sample, class_prototype_matrix, batch)

        return correlations

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

    @staticmethod
    def map_pred_labels_to_global(pred_labels, batch_global_labels, num_episodes):
        """

        :param pred_labels: (B*N*Q)
        :param batch_global_labels: (B*N)
        :param num_episodes: number of episodes in the batch

        :return
        """
        global_labels_per_episode = batch_global_labels.reshape(num_episodes, -1)
        pred_labels = pred_labels.reshape(num_episodes, -1)

        mapped_labels = []
        for episode_num in range(num_episodes):

            # shape (N)
            episode_global_labels = global_labels_per_episode[episode_num]
            # shape (N*Q)
            episode_pred_labels = pred_labels[episode_num]
            # shape (N*Q)
            episode_mapped_labels = episode_global_labels[episode_pred_labels]

            mapped_labels.append(episode_mapped_labels)

        # shape (B*N*Q)
        mapped_labels = torch.cat(mapped_labels, dim=0)

        return mapped_labels

    @abc.abstractmethod
    def compute_classification_loss(self, embedded_queries, class_prototypes, batch: EpisodeBatch, **kwargs):
        pass


class MolecularPrototypeBased(PrototypeBased):
    def __init__(self, loss_weights):
        super().__init__(loss_weights)
        self.mixup_augmentor = MolecularMixUpAugmentor(self)

    @staticmethod
    def map_pred_labels_to_active_or_not(pred_labels, batch_active_or_not_labels, num_episodes):
        """

        :param pred_labels: (B*N*Q)
        :param batch_active_or_not_labels: (B*N)
        :param num_episodes: number of episodes in the batch

        :return
        """
        global_labels_per_episode = batch_active_or_not_labels.reshape(num_episodes, -1)
        pred_labels = pred_labels.reshape(num_episodes, -1)

        mapped_labels = []
        for episode_num in range(num_episodes):

            # shape (N)
            episode_global_labels = global_labels_per_episode[episode_num]
            # shape (N*Q)
            episode_pred_labels = pred_labels[episode_num]
            # shape (N*Q)
            episode_mapped_labels = episode_global_labels[episode_pred_labels]

            mapped_labels.append(episode_mapped_labels)

        # shape (B*N*Q)
        mapped_labels = torch.cat(mapped_labels, dim=0)

        return mapped_labels
