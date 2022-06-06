import torch
from torch_geometric.data import Batch

from fs_grl.data.episode.episode_batch import EpisodeBatch
from fs_grl.modules.components.task_embedding_network import TaskEmbeddingNetwork


class TADAM:
    """
    Extension of the standard Prototypical Network architecture inspired by
    Task Adaptive Task Dependent Adaptive Metric https://arxiv.org/abs/1805.10123
    """

    def __init__(self, gamma_0_init: float, beta_0_init: float, **kwargs):

        self.task_embedding_network = TaskEmbeddingNetwork(
            hidden_size=self.embedder.embedding_dim // 2,
            embedding_dim=self.embedder.embedding_dim,
            num_convs=self.embedder.node_embedder.num_convs + 1,
            beta_0_init=beta_0_init,
            gamma_0_init=gamma_0_init,
        )

    def align_gammas_betas(
        self, gammas: torch.Tensor, betas: torch.Tensor, num_sample_repetitions: torch.Tensor, batch: Batch
    ):
        """

        :param gammas:
        :param betas:
        :param num_sample_repetitions:
        :param batch:

        :return
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

        :return tensor ~ (num_episodes, embedding_dim)
        """
        embedded_supports = self.embed_supports(batch)

        # list (num_episodes) of dicts {label: prototype, ...}
        prototypes_dicts = self.compute_prototypes(embedded_supports, batch)

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

        :return
        """
        return self.embedder(supports, gammas, betas)

    def task_conditioned_embed_queries(self, queries, gammas, betas):
        """
        Embed the queries accounting for the task embedding

        :param queries:
        :param gammas:
        :param betas:

        :return
        """
        return self.embedder(queries, gammas, betas)

    def compute_losses(self, model_out, batch):
        losses = super().compute_losses(model_out, batch)
        losses["film_reg"] = self.loss_weights["film_reg"] * (
            torch.norm(self.task_embedding_network.gamma_0, p=2) + torch.norm(self.task_embedding_network.beta_0, p=2)
        )
        losses["total"] = self.compute_total_loss(losses)
        return losses
