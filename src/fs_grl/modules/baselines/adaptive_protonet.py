import torch

from fs_grl.data.episode import EpisodeBatch
from fs_grl.modules.attention import MultiHeadAttention
from fs_grl.modules.baselines.prototypical_network import PrototypicalNetwork
from fs_grl.modules.similarities.squared_l2 import squared_l2


class AdaptivePrototypicalNetwork(PrototypicalNetwork):
    def __init__(
        self,
        cfg,
        feature_dim,
        num_classes,
        loss_weights,
        metric_scaling_factor,
        num_attention_heads,
        attention_dropout,
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
        :return:
        """

        protonet_out = super().forward(batch)

        prototypes_dicts, embedded_queries, embedded_supports = (
            protonet_out["prototypes_dicts"],
            protonet_out["embedded_queries"],
            protonet_out["embedded_supports"],
        )

        adapted_prototypes_dicts = self.adapt_prototypes(prototypes_dicts, batch)

        auxiliary_distances = self.get_auxiliary_distances(embedded_queries, embedded_supports, batch)

        distances = self.get_queries_prototypes_correlations_batch(embedded_queries, adapted_prototypes_dicts, batch)
        distances = self.metric_scaling_factor * distances

        return {
            "embedded_queries": embedded_queries,
            "embedded_supports": embedded_supports,
            "prototypes_dicts": adapted_prototypes_dicts,
            "distances": distances,
            "aux_distances": auxiliary_distances,
        }

    def get_auxiliary_distances(self, embedded_queries, embedded_supports, batch):
        """

        :param embedded_queries: (B*N*Q, E)
        :param embedded_supports: (B*N*K, E)
        :param batch:
        :return:
        """

        B, N, E = batch.num_episodes, batch.episode_hparams.num_classes_per_episode, self.embedder.embedding_dim
        K, Q = batch.episode_hparams.num_supports_per_class, batch.episode_hparams.num_queries_per_class

        # ( B, (N*Q)+(N*K) )
        local_labels = torch.cat(
            (batch.supports.support_local_labels.view(B, N * K), batch.queries.query_local_labels.view(B, N * Q)), dim=1
        )

        supports_reshaped = embedded_supports.view(B, N * K, E)
        queries_reshaped = embedded_queries.view(B, N * Q, E)

        # ( B, (N*K)+(N*Q), E )
        supports_and_queries = torch.cat((supports_reshaped, queries_reshaped), dim=1)

        # sort the embeddings by class for each episode
        local_labels_sorted = local_labels.argsort(dim=-1)

        # ( B, (N*K)+(N*Q), E )
        supports_and_queries_by_label = supports_and_queries[torch.arange(B).unsqueeze(-1), local_labels_sorted]

        # (B*N, K+Q, E)
        supports_and_queries_by_label = supports_and_queries_by_label.view(B * N, K + Q, E)

        # (B*N, K+Q, E)
        adapted_samples = self.attention(
            supports_and_queries_by_label, supports_and_queries_by_label, supports_and_queries_by_label
        )

        # (B, N, K + Q, E)
        adapted_samples = adapted_samples.view(B, N, K + Q, E)

        # (B, N, E)
        auxiliary_prototypes = torch.mean(adapted_samples, dim=2)

        # ( K+Q, B*N, E )
        supports_and_queries_by_label = supports_and_queries_by_label.permute([1, 0, 2])

        # ( B*N * (K+Q), 1, E )
        supports_and_queries_by_label = supports_and_queries_by_label.reshape(-1, E).unsqueeze(1)

        # (B, 1, N, E)
        auxiliary_prototypes = auxiliary_prototypes.unsqueeze(1)
        # (B, N*(K+Q), N, E)
        auxiliary_prototypes = auxiliary_prototypes.expand(B, N * (K + Q), N, E)

        # (B * N*(K+Q), N, E)
        auxiliary_prototypes = auxiliary_prototypes.reshape(B * N * (K + Q), N, E)

        # (B * N * (K + Q), N)
        auxiliary_distances = self.metric_scaling_factor * squared_l2(
            auxiliary_prototypes, supports_and_queries_by_label
        )

        return auxiliary_distances

    def adapt_prototypes(self, prototypes_dicts, batch):

        stacked_prototypes = torch.stack(
            [torch.stack(tuple(episode_prototype_dict.values())) for episode_prototype_dict in prototypes_dicts]
        )

        B, N, E = batch.num_episodes, batch.episode_hparams.num_classes_per_episode, self.embedder.embedding_dim
        assert list(stacked_prototypes.shape) == [B, N, E]

        adapted_prototypes = self.attention(stacked_prototypes, stacked_prototypes, stacked_prototypes)

        adapted_prototypes_dict = [
            {lab: prot for lab, prot in zip(episode_prototype_dict.keys(), episode_adapted_prots)}
            for episode_adapted_prots, episode_prototype_dict in zip(adapted_prototypes, prototypes_dicts)
        ]
        return adapted_prototypes_dict

    def compute_losses(self, model_out, batch):
        losses = super().compute_losses(model_out, batch)

        losses["adaptive_reg"] = self.compute_adaptive_reg(model_out, batch)

        losses["total"] = self.compute_total_loss(losses)

        return losses

    def compute_adaptive_reg(self, model_out, batch):
        B, N, K, Q = (
            batch.num_episodes,
            batch.episode_hparams.num_classes_per_episode,
            batch.episode_hparams.num_supports_per_class,
            batch.episode_hparams.num_queries_per_class,
        )

        # (B * ( (N * K) + (N * Q) ), N)
        distances = model_out["aux_distances"]
        assert list(distances.shape) == [B * ((N * K) + (N * Q)), N]

        # (B, (N*K) + (N*Q), N)
        distances = distances.view(B, (N * K) + (N * Q), N)
        logits = -distances

        # (B, (N*Q))
        query_labels = batch.queries.query_local_labels.view(B, N * Q)
        # (B, (N*K))
        support_labels = batch.supports.support_local_labels.view(B, N * K)

        # (B, (N*K) + (N*Q))
        labels_per_episode = torch.cat((support_labels, query_labels), dim=1)
        labels_per_episode = torch.sort(labels_per_episode, dim=-1).values

        cum_loss = 0
        for episode in range(batch.num_episodes):
            cum_loss += self.loss_func(logits[episode], labels_per_episode[episode])

        cum_loss /= batch.num_episodes
        return cum_loss
