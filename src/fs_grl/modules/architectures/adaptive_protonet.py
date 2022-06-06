from typing import Dict, List

import torch
from omegaconf import DictConfig

from fs_grl.data.episode.episode_batch import EpisodeBatch
from fs_grl.modules.architectures.protonet import PrototypicalNetwork
from fs_grl.modules.components.attention import MultiHeadAttention
from fs_grl.modules.similarities.squared_l2 import squared_l2


class AdaptivePrototypicalNetwork(PrototypicalNetwork):
    """
    Extension of the standard Prototypical Network architecture inspired by
    Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions https://arxiv.org/abs/1812.03664
    """

    def __init__(
        self,
        cfg: DictConfig,
        feature_dim: int,
        num_classes: int,
        loss_weights: Dict,
        metric_scaling_factor: float,
        num_attention_heads: int,
        attention_dropout: float,
        **kwargs
    ):
        super().__init__(
            cfg,
            feature_dim=feature_dim,
            num_classes=num_classes,
            metric_scaling_factor=metric_scaling_factor,
            loss_weights=loss_weights,
            **kwargs,
        )

        self.attention = MultiHeadAttention(
            n_head=num_attention_heads,
            d_model=self.embedder.embedding_dim,
            d_k=self.embedder.embedding_dim,
            d_v=self.embedder.embedding_dim,
            dropout=attention_dropout,
        )

    def forward(self, batch: EpisodeBatch):
        """
        :param batch:

        :return
        """

        # samples are embedded as in the standard ProtoNet
        protonet_out = super().forward(batch)

        prototypes_dicts, embedded_queries, embedded_supports = (
            protonet_out["prototypes_dicts"],
            protonet_out["embedded_queries"],
            protonet_out["embedded_supports"],
        )

        adapted_prototypes_dicts = self.adapt_prototypes(prototypes_dicts, batch)

        auxiliary_distances = self.compute_auxiliary_distances(embedded_queries, embedded_supports, batch)

        distances = self.compute_queries_prototypes_correlations_batch(
            embedded_queries, adapted_prototypes_dicts, batch
        )
        distances = self.metric_scaling_factor * distances

        return {
            "embedded_queries": embedded_queries,
            "embedded_supports": embedded_supports,
            "prototypes_dicts": adapted_prototypes_dicts,
            "distances": distances,
            "aux_distances": auxiliary_distances,
        }

    def adapt_prototypes(self, prototypes_dicts: List[Dict], batch: EpisodeBatch) -> List[Dict]:
        """
        Adapt prototypes to the specific episode by passing them into a Transformer block

        :param prototypes_dicts: list (num_episodes) containing for each episode the mapping
                                 label -> prototype
        :param batch:

        :return: adapted prototypes for each episode
        """

        stacked_prototypes = torch.stack(
            [torch.stack(tuple(episode_prototype_dict.values())) for episode_prototype_dict in prototypes_dicts]
        )

        B, N, E = (
            batch.num_episodes,
            batch.episode_hparams.num_classes_per_episode,
            self.embedder.embedding_dim,
        )
        assert list(stacked_prototypes.shape) == [B, N, E]

        adapted_prototypes = self.attention(stacked_prototypes, stacked_prototypes, stacked_prototypes)

        adapted_prototypes_dicts = []
        for episode_adapted_prots, episode_prototype_dict in zip(adapted_prototypes, prototypes_dicts):
            episode_labels = episode_prototype_dict.keys()
            episode_adapted_proto_dict = {}

            for label, adapted_prototype in zip(episode_labels, episode_adapted_prots):
                episode_adapted_proto_dict[label] = adapted_prototype

            adapted_prototypes_dicts.append(episode_adapted_proto_dict)

        return adapted_prototypes_dicts

    def compute_auxiliary_distances(
        self, embedded_queries: torch.Tensor, embedded_supports: torch.Tensor, batch: EpisodeBatch
    ):
        """
        # TODO: validate and refactor

        :param embedded_queries: (B*N*Q, E)
        :param embedded_supports: (B*N*K, E)
        :param batch:

        :return
        """

        B, N, E = (
            batch.num_episodes,
            batch.episode_hparams.num_classes_per_episode,
            self.embedder.embedding_dim,
        )
        K, Q = batch.episode_hparams.num_supports_per_class, batch.episode_hparams.num_queries_per_class

        num_supports_episode = batch.episode_hparams.num_supports_per_episode
        num_queries_episode = batch.episode_hparams.num_queries_per_episode
        num_samples_label = K + Q

        # (B, num_supports_episode)
        support_local_y_by_episode = batch.supports.local_y.view(B, num_supports_episode)
        # (B, num_queries_episode)
        query_local_y_by_episode = batch.queries.local_y.view(B, num_queries_episode)

        # (B, total_samples_episode)
        local_labels = torch.cat(
            (support_local_y_by_episode, query_local_y_by_episode),
            dim=1,
        )

        supports_by_episode = embedded_supports.view(B, num_supports_episode, E)
        queries_by_episode = embedded_queries.view(B, num_queries_episode, E)

        # ( B, total_samples_episode, E )
        samples = torch.cat((supports_by_episode, queries_by_episode), dim=1)

        # sort the embeddings by class for each episode
        local_labels_sorted = local_labels.argsort(dim=-1)

        # ( B, total_samples_episode, E )
        samples_sorted_by_label = samples[torch.arange(B).unsqueeze(-1), local_labels_sorted]

        # (B*N, num_samples_label, E)
        samples_sorted_by_label = samples_sorted_by_label.view(B * N, num_samples_label, E)

        # (B*N, num_samples_label, E)
        adapted_samples = self.attention(samples_sorted_by_label, samples_sorted_by_label, samples_sorted_by_label)

        # (B, N, num_samples_label, E)
        adapted_samples = adapted_samples.view(B, N, num_samples_label, E)

        # ( B * N * (num_samples_label), 1, E )
        samples_sorted_by_label = (
            samples_sorted_by_label.reshape(-1, E).unsqueeze(1).expand(B * N * num_samples_label, N, E)
        )

        # (B, N, E)
        auxiliary_prototypes = torch.mean(adapted_samples, dim=2)

        # (B, 1, N, E)
        auxiliary_prototypes = auxiliary_prototypes.unsqueeze(1)
        # (B, N*(num_samples_label), N, E)
        auxiliary_prototypes = auxiliary_prototypes.expand(B, N * num_samples_label, N, E)

        # (B * N * (num_samples_label), N, E)
        auxiliary_prototypes = auxiliary_prototypes.reshape(B * N * num_samples_label, N, E)

        # (B * N * (num_samples_label), N)
        auxiliary_distances = self.metric_scaling_factor * squared_l2(auxiliary_prototypes, samples_sorted_by_label)

        return auxiliary_distances

    def compute_losses(self, model_out, batch):
        losses = super().compute_losses(model_out, batch)

        if self.loss_weights["adaptive_reg"] > 0:
            losses["adaptive_reg"] = self.compute_adaptive_reg(model_out, batch)

        losses["total"] = self.compute_total_loss(losses)

        return losses

    def compute_adaptive_reg(self, model_out, batch):
        """

        :param model_out:
        :param batch:
        :return:
        """
        B, N, K, Q = (
            batch.num_episodes,
            batch.episode_hparams.num_classes_per_episode,
            batch.episode_hparams.num_supports_per_class,
            batch.episode_hparams.num_queries_per_class,
        )
        num_supports_episode, num_queries_episode = (
            batch.episode_hparams.num_supports_per_episode,
            batch.episode_hparams.num_queries_per_episode,
        )
        num_samples_label = K + Q

        # (B * N * num_samples_label, N)
        distances = model_out["aux_distances"]
        assert list(distances.shape) == [B * N * num_samples_label, N]

        # (B, N * num_samples_label, N)
        distances = distances.view(B, N * num_samples_label, N)
        logits = -distances

        # (B, num_queries_episode)
        query_local_y_by_episode = batch.queries.local_y.view(B, num_queries_episode)

        # (B, num_supports_episode)
        support_local_y_by_episode = batch.supports.local_y.view(B, num_supports_episode)

        # (B, num_queries_episode + num_supports_episode)
        labels_per_episode = torch.cat((support_local_y_by_episode, query_local_y_by_episode), dim=1)
        labels_per_episode = torch.sort(labels_per_episode, dim=-1).values

        cum_loss = 0
        for episode in range(batch.num_episodes):
            cum_loss += self.loss_func(logits[episode], labels_per_episode[episode])

        cum_loss /= batch.num_episodes
        return cum_loss
