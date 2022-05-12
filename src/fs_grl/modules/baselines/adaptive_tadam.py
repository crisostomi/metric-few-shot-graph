import torch

from fs_grl.data.episode import EpisodeBatch
from fs_grl.modules.attention import MultiHeadAttention
from fs_grl.modules.baselines.tadam import TADAM
from fs_grl.modules.similarities.squared_l2 import squared_l2
from fs_grl.modules.task_embedding_network import TaskEmbeddingNetwork


class AdaptiveTADAM(TADAM):
    def __init__(
        self,
        cfg,
        feature_dim,
        num_classes,
        loss_weights,
        metric_scaling_factor,
        gamma_0_init,
        beta_0_init,
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
            gamma_0_init=gamma_0_init,
            beta_0_init=beta_0_init,
            **kwargs,
        )

        self.TEN = TaskEmbeddingNetwork(
            in_size=self.embedder.embedding_dim,
            hidden_size=self.embedder.embedding_dim // 2,
            embedding_dim=self.embedder.embedding_dim,
            num_convs=self.embedder.node_embedder.num_convs,
            beta_0_init=beta_0_init,
            gamma_0_init=gamma_0_init,
        )

        self.attention = MultiHeadAttention(
            n_head=num_attention_heads,
            d_model=self.embedder.embedding_dim,
            d_k=self.embedder.embedding_dim,
            d_v=self.embedder.embedding_dim,
            dropout=attention_dropout,
        )

    def forward(self, batch: EpisodeBatch):

        episode_embeddings = self.get_episode_embeddings(batch)
        gammas, betas = self.TEN(episode_embeddings)

        num_supports_repetitions = (
            torch.tensor([batch.num_supports_per_episode for _ in range(batch.num_episodes)]).type_as(gammas).int()
        )
        num_queries_repetitions = (
            torch.tensor([batch.num_queries_per_episode for _ in range(batch.num_episodes)]).type_as(gammas).int()
        )

        support_gammas = torch.repeat_interleave(gammas, num_supports_repetitions, dim=0)
        support_betas = torch.repeat_interleave(betas, num_supports_repetitions, dim=0)

        query_gammas = torch.repeat_interleave(gammas, num_queries_repetitions, dim=0)
        query_betas = torch.repeat_interleave(betas, num_queries_repetitions, dim=0)

        embedded_supports = self.task_conditioned_embed_supports(batch.supports, support_gammas, support_betas)
        embedded_queries = self.task_conditioned_embed_queries(batch.queries, query_gammas, query_betas)
        prototypes_dicts = self.get_prototypes(embedded_supports, batch)

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

    def get_episode_embeddings(self, batch):
        graph_level = not self.prototypes_from_nodes
        embedded_supports = self.embed_supports(batch, graph_level=graph_level)

        # list (num_episodes) of dicts {label: prototype, ...}
        prototypes_dicts = self.get_prototypes(embedded_supports, batch)

        episode_embeddings = []
        for prototypes_dict_per_episode in prototypes_dicts:
            episode_embedding = torch.stack([prototype for prototype in prototypes_dict_per_episode.values()]).mean(
                dim=0
            )
            episode_embeddings.append(episode_embedding)

        episode_embeddings = torch.stack(episode_embeddings)
        return episode_embeddings

    def task_conditioned_embed_supports(self, supports, gammas, betas):
        return self.embedder(supports, gammas, betas)

    def task_conditioned_embed_queries(self, queries, gammas, betas):
        return self.embedder(queries, gammas, betas)

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

    def compute_losses(self, model_out, batch):
        losses = super().compute_losses(model_out, batch)

        if self.loss_weights["adaptive_reg"] > 0:
            losses["adaptive_reg"] = self.compute_adaptive_reg(model_out, batch)

        losses["total"] = self.compute_total_loss(losses)

        return losses

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
