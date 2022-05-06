import torch

from fs_grl.data.episode import EpisodeBatch
from fs_grl.modules.baselines.prototypical_network import PrototypicalNetwork
from fs_grl.modules.task_embedding_network import TaskEmbeddingNetwork


class TADAM(PrototypicalNetwork):
    def __init__(self, cfg, feature_dim, num_classes, loss_weights, metric_scaling_factor, gamma_0, beta_0, **kwargs):
        super().__init__(
            cfg,
            feature_dim=feature_dim,
            num_classes=num_classes,
            metric_scaling_factor=metric_scaling_factor,
            loss_weights=loss_weights,
            **kwargs,
        )

        self.TEN = TaskEmbeddingNetwork(
            in_size=self.embedder.embedding_dim,
            hidden_size=self.embedder.embedding_dim // 2,
            out_size=self.embedder.node_embedder.num_convs,
            beta_0=beta_0,
            gamma_0=gamma_0,
        )

    def forward(self, batch: EpisodeBatch):

        episode_embeddings = self.get_episode_embeddings(batch)
        gammas, betas = self.TEN(episode_embeddings)

        support_gammas = torch.repeat_interleave(
            gammas, torch.tensor([batch.num_supports_per_episode for _ in range(batch.num_episodes)]), dim=0
        )
        support_betas = torch.repeat_interleave(
            betas, torch.tensor([batch.num_supports_per_episode for _ in range(batch.num_episodes)]), dim=0
        )

        query_gammas = torch.repeat_interleave(
            gammas, torch.tensor([batch.num_queries_per_episode for _ in range(batch.num_episodes)]), dim=0
        )
        query_betas = torch.repeat_interleave(
            betas, torch.tensor([batch.num_queries_per_episode for _ in range(batch.num_episodes)]), dim=0
        )

        embedded_supports = self.task_conditioned_embed_supports(batch.supports, support_gammas, support_betas)
        embedded_queries = self.task_conditioned_embed_queries(batch.queries, query_gammas, query_betas)
        prototypes_dicts = self.get_prototypes(embedded_supports, batch)

        distances = self.get_queries_prototypes_correlations_batch(embedded_queries, prototypes_dicts, batch)
        distances = self.metric_scaling_factor * distances

        return {
            "embedded_queries": embedded_queries,
            "embedded_supports": embedded_supports,
            "prototypes_dicts": prototypes_dicts,
            "distances": distances,
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
