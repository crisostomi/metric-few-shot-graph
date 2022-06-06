import abc
from typing import Dict

import torch

from fs_grl.data.episode.episode_batch import CosineEpisodeBatch, EpisodeBatch
from fs_grl.modules.architectures.gnn_prototype_based import GNNPrototypeBased


class GNNEmbeddingPairwise(GNNPrototypeBased, abc.ABC):
    def __init__(self, cfg, feature_dim, num_classes, **kwargs):
        super().__init__(cfg, feature_dim=feature_dim, num_classes=num_classes, **kwargs)

    def compute_classification_loss(self, model_out, batch: CosineEpisodeBatch, **kwargs):
        similarities = model_out["similarities"]

        return self.loss_func(similarities, batch.cosine_targets)

    def get_predictions(self, step_out: Dict, batch: EpisodeBatch) -> torch.Tensor:
        """

        :param similarities: shape (B * N*Q * N)
        :param batch:

        :return
        """
        # shape ~(num_episodes * num_queries_per_class * num_classes_per_episode)
        similarities = step_out["similarities"]

        num_classes_per_episode = batch.episode_hparams.num_classes_per_episode

        # shape (B*(N*Q), N) contains the similarity between the query
        # and the N label prototypes for each of the N*Q queries
        similarities_per_label = similarities.reshape((-1, num_classes_per_episode))

        # shape (B*(N*Q)) contains for each query the most similar label
        pred_labels = torch.argmax(similarities_per_label, dim=-1)

        pred_global_labels = self.map_pred_labels_to_global(
            pred_labels=pred_labels, batch_global_labels=batch.global_labels, num_episodes=batch.num_episodes
        )

        return pred_global_labels
