import torch
from torch_geometric.data import Batch

from fs_grl.data.episode import EpisodeBatch
from fs_grl.modules.baselines.prototypical_network import PrototypicalNetwork
from fs_grl.modules.task_embedding_network import TaskEmbeddingNetwork


class TADAM(PrototypicalNetwork):
    """
    Extension of the standard Prototypical Network architecture inspired by https://arxiv.org/abs/1805.10123
    """

    def __init__(
        self, cfg, feature_dim, num_classes, loss_weights, metric_scaling_factor, gamma_0_init, beta_0_init, **kwargs
    ):
        super().__init__(
            cfg,
            feature_dim=feature_dim,
            num_classes=num_classes,
            metric_scaling_factor=metric_scaling_factor,
            loss_weights=loss_weights,
            **kwargs
        )

        self.task_embedding_network = TaskEmbeddingNetwork(
            hidden_size=self.embedder.embedding_dim // 2,
            embedding_dim=self.embedder.embedding_dim,
            num_convs=self.embedder.node_embedder.num_convs + 1,
            beta_0_init=beta_0_init,
            gamma_0_init=gamma_0_init,
        )

    def forward(self, batch: EpisodeBatch):

        episode_embeddings = self.get_episode_embeddings(batch)
        gammas, betas = self.task_embedding_network(episode_embeddings)

        num_supports_repetitions = (
            torch.tensor([batch.num_supports_per_episode for _ in range(batch.num_episodes)]).type_as(gammas).int()
        )
        num_queries_repetitions = (
            torch.tensor([batch.num_queries_per_episode for _ in range(batch.num_episodes)]).type_as(gammas).int()
        )

        query_gammas, query_betas = self.align_gammas_betas(gammas, betas, num_queries_repetitions, batch.queries)
        support_gammas, support_betas = self.align_gammas_betas(gammas, betas, num_supports_repetitions, batch.supports)

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

    def align_gammas_betas(
        self, gammas: torch.Tensor, betas: torch.Tensor, num_sample_repetitions: torch.Tensor, batch: Batch
    ):
        """

        :param gammas:
        :param betas:
        :param num_sample_repetitions:
        :param batch:
        :return:
        """

        # (num_samples_in_batch, embedding_dim, num_convs)
        gammas_repeated_by_graphs = torch.repeat_interleave(gammas, num_sample_repetitions, dim=0)
        betas_repeated_by_graphs = torch.repeat_interleave(betas, num_sample_repetitions, dim=0)

        _, num_repetitions_nodes = torch.unique(batch.batch, return_counts=True)

        # (num_nodes_in_batch, embedding_dim, num_convs)
        gammas_repeated_by_nodes = torch.repeat_interleave(gammas_repeated_by_graphs, num_repetitions_nodes, dim=0)
        betas_repeated_by_nodes = torch.repeat_interleave(betas_repeated_by_graphs, num_repetitions_nodes, dim=0)

        # (num_convs, num_nodes_in_batch, embedding_dim)
        return gammas_repeated_by_nodes.permute(2, 0, 1), betas_repeated_by_nodes.permute(2, 0, 1)

    def get_episode_embeddings(self, batch: EpisodeBatch) -> torch.Tensor:
        """
        Get the episode representation as the mean of the class prototypes of the episode

        :param batch:

        :return: tensor ~ (num_episodes, embedding_dim)
        """
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

    def task_conditioned_embed_supports(self, supports: Batch, gammas: torch.Tensor, betas: torch.Tensor):
        """
        Embed the supports accounting for the task embedding

        :param supports:
        :param gammas:
        :param betas:

        :return:
        """
        return self.embedder(supports, gammas, betas)

    def task_conditioned_embed_queries(self, queries, gammas, betas):
        """
        Embed the queries accounting for the task embedding

        :param queries:
        :param gammas:
        :param betas:

        :return:
        """
        return self.embedder(queries, gammas, betas)

    def compute_losses(self, model_out, batch):
        losses = super().compute_losses(model_out, batch)
        losses["film_reg"] = self.loss_weights["film_reg"] * (
            torch.norm(self.task_embedding_network.gamma_0, p=2) + torch.norm(self.task_embedding_network.beta_0, p=2)
        )
        losses["total"] = self.compute_total_loss(losses)
        return losses
