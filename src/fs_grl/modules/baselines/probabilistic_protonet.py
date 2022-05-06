import torch
from torch import nn

from fs_grl.data.episode import EpisodeBatch
from fs_grl.modules.baselines.prototypical_network import PrototypicalNetwork


class ProbabilisticPrototypicalNetwork(PrototypicalNetwork):
    def __init__(self, cfg, feature_dim, num_classes, loss_weights, metric_scaling_factor, **kwargs):
        super().__init__(
            cfg,
            feature_dim=feature_dim,
            num_classes=num_classes,
            metric_scaling_factor=metric_scaling_factor,
            loss_weights=loss_weights,
            **kwargs,
        )

        self.linear_mean = nn.Linear(in_features=self.embedder.embedding_dim, out_features=self.embedder.embedding_dim)
        self.linear_logvar = nn.Linear(
            in_features=self.embedder.embedding_dim, out_features=self.embedder.embedding_dim
        )
        self.supports_aggregation = "probabilistic"

    def forward(self, batch: EpisodeBatch):
        """
        :param batch:
        :return:
        """

        graph_level = not self.prototypes_from_nodes

        # (num_supports_batch, embedding_dim)
        embedded_supports = self.embed_supports(batch, graph_level=graph_level)

        # (num_supports_batch, embedding_dim*2)
        supports_mean_logvar = self.embed_mean_logvar(embedded_supports)

        # shape (num_queries_batch, embedding_dim)
        embedded_queries = self.embed_queries(batch)

        # list (num_episodes) of dicts {label: prototype, ...}
        prototypes_dicts = self.get_prototypes(supports_mean_logvar, batch)

        distances = self.get_queries_prototypes_correlations_batch(embedded_queries, prototypes_dicts, batch)
        distances = self.metric_scaling_factor * distances

        return {
            "embedded_queries": embedded_queries,
            "embedded_supports": embedded_supports,
            "prototypes_dicts": prototypes_dicts,
            "distances": distances,
            "supports_mean_logvar": supports_mean_logvar,
        }

    def embed_mean_logvar(self, supports: torch.Tensor) -> torch.Tensor:
        """

        :param supports: (num_supports_batch, embedding_dim)
        :return:
        """
        mean = self.linear_mean(supports)
        logvar = self.linear_logvar(supports)

        return torch.cat((mean, logvar), dim=-1)

    def probabilistic_embed(self, supports: torch.Tensor) -> torch.Tensor:
        """

        :param supports: (num_supports_per_class, embedding_dim)

        :return:
        """

        mean_log_var = supports.mean(dim=0)

        E = self.embedder.embedding_dim
        mean, log_var = torch.split(mean_log_var, E, dim=0)

        prototype = self.latent_sample(mean, log_var)
        return prototype

    def latent_sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """

        :param mu:
        :param logvar:

        :return:
        """

        if self.training:
            # Convert the logvar to std
            std = (logvar * 0.5).exp()

            # the reparameterization trick
            return torch.distributions.Normal(loc=mu, scale=std).rsample()
        else:
            return mu

    def compute_losses(self, model_out, batch):
        losses = super().compute_losses(model_out, batch)

        # losses["KL_reg"] = self.compute_KL_reg(model_out, batch)

        losses["total"] = self.compute_total_loss(losses)

        return losses

    def compute_KL_reg(self, model_out, batch):

        # (num_supports_batch, embedding_dim*2)
        mean_logvar = model_out["supports_mean_logvar"]
        mean, log_var = torch.split(mean_logvar, self.embedder.embedding_dim, dim=-1)

        kldivergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return kldivergence
