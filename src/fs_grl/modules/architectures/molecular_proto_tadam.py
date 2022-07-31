from typing import Dict

import torch
from omegaconf import DictConfig

from fs_grl.data.episode.episode_batch import MolecularEpisodeBatch
from fs_grl.modules.architectures.molecular_protonet import MolecularPrototypicalNetwork
from fs_grl.modules.architectures.tadam import TADAM


class MolecularProtoTADAM(TADAM, MolecularPrototypicalNetwork):
    """
    Extension of the standard Prototypical Network architecture inspired by
    Task Adaptive Task Dependent Adaptive Metric https://arxiv.org/abs/1805.10123
    """

    def __init__(
        self,
        cfg: DictConfig,
        feature_dim: int,
        num_classes: int,
        loss_weights: Dict,
        metric_scaling_factor: float,
        gamma_0_init: float,
        beta_0_init: float,
        **kwargs
    ):
        MolecularPrototypicalNetwork.__init__(
            self,
            cfg,
            feature_dim=feature_dim,
            num_classes=num_classes,
            metric_scaling_factor=metric_scaling_factor,
            loss_weights=loss_weights,
            **kwargs
        )
        TADAM.__init__(self, gamma_0_init=gamma_0_init, beta_0_init=beta_0_init)

    def forward(self, batch: MolecularEpisodeBatch):

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

        distances = self.compute_queries_prototypes_correlations_batch(embedded_queries, prototypes_dicts, batch)
        distances = self.metric_scaling_factor * distances

        return {
            "embedded_queries": embedded_queries,
            "embedded_supports": embedded_supports,
            "prototypes_dicts": prototypes_dicts,
            "distances": distances,
        }

    def compute_losses(self, model_out, batch):
        losses = super().compute_losses(model_out, batch)

        losses["film_reg"] = self.loss_weights["film_reg"] * (
            torch.norm(self.task_embedding_network.gamma_0, p=2) + torch.norm(self.task_embedding_network.beta_0, p=2)
        )

        losses["total"] = self.compute_total_loss(losses)
        return losses
