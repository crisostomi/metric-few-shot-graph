from typing import Dict, List

import torch
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss

from fs_grl.data.episode.episode_batch import EpisodeBatch
from fs_grl.modules.architectures.gnn_embedding_similarity import GNNEmbeddingSimilarity
from fs_grl.modules.similarities.squared_l2 import squared_l2


class PrototypicalNetwork(GNNEmbeddingSimilarity):
    def __init__(
        self,
        cfg: DictConfig,
        feature_dim: int,
        num_classes: int,
        metric_scaling_factor: float,
        loss_weights: Dict,
        **kwargs
    ):
        super().__init__(
            cfg,
            feature_dim=feature_dim,
            num_classes=num_classes,
            metric_scaling_factor=metric_scaling_factor,
            loss_weights=loss_weights,
            **kwargs
        )
        self.loss_func = CrossEntropyLoss()

    def forward(self, batch: EpisodeBatch):
        """
        :param batch:
        :return
        """

        graph_level = not self.prototypes_from_nodes
        embedded_supports = self.embed_supports(batch, graph_level=graph_level)

        # shape (num_queries_batch, hidden_dim)
        embedded_queries = self.embed_queries(batch)

        # list (num_episodes) of dicts {label: prototype, ...}
        prototypes_dicts = self.get_prototypes(embedded_supports, batch)

        distances = self.get_queries_prototypes_correlations_batch(embedded_queries, prototypes_dicts, batch)
        distances = self.metric_scaling_factor * distances

        return {
            "embedded_queries": embedded_queries,
            "embedded_supports": embedded_supports,
            "prototypes_dicts": prototypes_dicts,
            "distances": distances,
        }

    def get_queries_prototypes_correlations_batch(
        self, embedded_queries: torch.Tensor, class_prototypes: List[Dict], batch: EpisodeBatch
    ):
        """

        :param embedded_queries ~ (num_queries_batch*num_classes, hidden_dim)
        :param class_prototypes ~ (num_queries_batch*num_classes, hidden_dim)
        :param batch:

        :return
        """
        batch_queries_prototypes = self.align_queries_prototypes(batch, embedded_queries, class_prototypes)
        batch_queries, batch_prototypes = batch_queries_prototypes["queries"], batch_queries_prototypes["prototypes"]

        distances = squared_l2(batch_queries, batch_prototypes)

        return distances

    def get_sample_prototypes_correlations(
        self, sample: torch.Tensor, prototypes: torch.Tensor, batch: EpisodeBatch
    ) -> torch.Tensor:
        """
        :param sample:
        :param prototypes:
        :param batch:

        :return
        """
        N = batch.episode_hparams.num_classes_per_episode
        repeated_sample = sample.repeat((N, 1))

        return squared_l2(repeated_sample, prototypes)

    def get_predictions(self, step_out: Dict, batch: EpisodeBatch) -> torch.Tensor:
        """

        :param step_out:
        :param batch:

        :return
        """
        # shape ~(num_episodes * num_queries_per_class * num_classes_per_episode)
        distances = step_out["model_out"]["distances"]

        distances = distances.view(batch.num_episodes, -1, batch.episode_hparams.num_classes_per_episode)
        logits = -distances

        probabilities = torch.log_softmax(logits, dim=-1)

        # shape (B*(N*Q)) contains for each query the most similar label
        pred_labels = torch.argmax(probabilities, dim=-1)

        pred_global_labels = self.map_pred_labels_to_global(
            pred_labels=pred_labels, batch_global_labels=batch.global_labels, num_episodes=batch.num_episodes
        )

        return pred_global_labels

    def get_sample_class_distribution(
        self, sample: torch.Tensor, label_to_prototype_embeddings: Dict, batch: EpisodeBatch
    ):
        """
        :param sample:
        :param label_to_prototype_embeddings:
        :param batch:
        :return
        """
        sampler_to_prototypes_distances = self.compute_sample_prototypes_correlations(
            sample, label_to_prototype_embeddings, batch
        )
        sample_class_distr = torch.softmax(-self.metric_scaling_factor * sampler_to_prototypes_distances, dim=-1)

        return sample_class_distr

    def compute_losses(self, model_out, batch):
        """

        :return
        """

        losses = {"classification_loss": 0, "latent_mixup_reg": 0, "intraclass_var_reg": 0}

        losses["classification_loss"] = self.compute_classification_loss(model_out, batch)

        if self.loss_weights["latent_mixup_reg"] > 0:
            losses["latent_mixup_reg"] = self.compute_latent_mixup_reg(model_out, batch)

        if self.loss_weights["intraclass_var_reg"] > 0:
            losses["intraclass_var_reg"] = self.model.compute_intraclass_var_reg(
                model_out["embedded_supports"], model_out["prototypes_dicts"], batch
            )

        losses["total"] = self.compute_total_loss(losses)

        return losses

    def compute_classification_loss(self, model_out: Dict, batch: EpisodeBatch, **kwargs):
        # shape (B, N*Q, N)
        distances = model_out["distances"]

        distances = distances.view(batch.num_episodes, -1, batch.episode_hparams.num_classes_per_episode)
        logits = -distances

        # TODO LOCAL_Y_BUG: fix, only one between query/support local labels and corresponding sample.local_y must remain

        # THIS WORKS
        local_labels_per_episode = batch.queries.query_local_labels.view(
            (batch.num_episodes, batch.episode_hparams.num_queries_per_episode)
        )

        # THIS DOESN'T
        local_labels_per_episode = batch.queries.local_y.view(
            (batch.num_episodes, batch.episode_hparams.num_queries_per_episode)
        )

        cum_loss = 0
        for episode in range(batch.num_episodes):
            cum_loss += self.loss_func(logits[episode], local_labels_per_episode[episode])

        cum_loss /= batch.num_episodes
        return cum_loss

    def compute_total_loss(self, losses):
        return sum(
            [
                loss_value * self.loss_weights[loss_name]
                for loss_name, loss_value in losses.items()
                if loss_name != "total"
            ]
        )
