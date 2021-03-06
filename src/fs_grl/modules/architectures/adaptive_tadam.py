import torch

from fs_grl.data.episode.episode_batch import EpisodeBatch
from fs_grl.modules.architectures.adaptive_protonet import AdaptivePrototypicalNetwork
from fs_grl.modules.architectures.tadam import TADAM
from fs_grl.modules.components.attention import MultiHeadAttention


class AdaptiveTADAM(TADAM, AdaptivePrototypicalNetwork):
    """
    Extension of the standard Prototypical Network that merges the task embedding adaption from
    "Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions" https://arxiv.org/abs/1812.03664
    and that of "Task Adaptive Task Dependent Adaptive Metric" https://arxiv.org/abs/1805.10123
    """

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
        AdaptivePrototypicalNetwork.__init__(
            self,
            cfg=cfg,
            feature_dim=feature_dim,
            num_classes=num_classes,
            metric_scaling_factor=metric_scaling_factor,
            loss_weights=loss_weights,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            **kwargs,
        )
        TADAM.__init__(self, gamma_0_init=gamma_0_init, beta_0_init=beta_0_init)

        self.attention = MultiHeadAttention(
            n_head=num_attention_heads,
            d_model=self.embedder.embedding_dim,
            d_k=self.embedder.embedding_dim,
            d_v=self.embedder.embedding_dim,
            dropout=attention_dropout,
        )

    def forward(self, batch: EpisodeBatch):

        episode_embeddings = self.get_episode_embeddings(batch)
        gammas, betas = self.task_embedding_network(episode_embeddings)

        num_supports_repetitions = (
            torch.tensor([batch.episode_hparams.num_supports_per_episode for _ in range(batch.num_episodes)])
            .type_as(gammas)
            .int()
        )
        num_queries_repetitions = (
            torch.tensor([batch.episode_hparams.num_queries_per_episode for _ in range(batch.num_episodes)])
            .type_as(gammas)
            .int()
        )

        query_gammas, query_betas = self.align_gammas_betas(gammas, betas, num_queries_repetitions, batch.queries)
        support_gammas, support_betas = self.align_gammas_betas(gammas, betas, num_supports_repetitions, batch.supports)

        embedded_supports = self.task_conditioned_embed_supports(batch.supports, support_gammas, support_betas)
        embedded_queries = self.task_conditioned_embed_queries(batch.queries, query_gammas, query_betas)
        prototypes_dicts = self.compute_prototypes(embedded_supports, batch)

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

    def compute_losses(self, model_out, batch):

        losses = super().compute_losses(model_out, batch)

        losses["film_reg"] = self.loss_weights["film_reg"] * (
            torch.norm(self.task_embedding_network.gamma_0, p=2) + torch.norm(self.task_embedding_network.beta_0, p=2)
        )

        if self.loss_weights["adaptive_reg"] > 0:
            losses["adaptive_reg"] = self.compute_adaptive_reg(model_out, batch)

        losses["total"] = self.compute_total_loss(losses)
        return losses
