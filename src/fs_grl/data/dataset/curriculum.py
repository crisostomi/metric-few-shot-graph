import logging
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data

from fs_grl.data.datamodule.curriculum_datamodule import GraphCurriculumDataModule
from fs_grl.data.dataset.episodic import IterableEpisodicDataset
from fs_grl.data.episode.episode import EpisodeHParams
from fs_grl.modules.similarities.squared_l2 import squared_l2

pylogger = logging.getLogger(__name__)


# TODO: get it back to work and refactor
class CurriculumIterableEpisodicDataset(IterableEpisodicDataset):
    def __init__(
        self,
        num_episodes: int,
        samples: List[Data],
        stage_labels: List[int],
        episode_hparams: EpisodeHParams,
        datamodule: GraphCurriculumDataModule,
    ):
        """

        :param num_episodes:
        :param samples:
        :param stage_labels:
        :param episode_hparams:
        :param datamodule:
        """
        super(CurriculumIterableEpisodicDataset, self).__init__(
            num_episodes=num_episodes,
            samples=samples,
            stage_labels=stage_labels,
            episode_hparams=episode_hparams,
        )

        self.datamodule = datamodule

        self.scaling_factor_path = "/".join(datamodule.prototypes_path.split("/")[:-1]) + "/scaling_factor.pt"

        class_prototypes: Dict[str, torch.Tensor] = torch.load(datamodule.prototypes_path)
        label_prototypes: Dict[int, torch.Tensor] = {
            datamodule.class_to_label_dict[cls]: cls_prototype for cls, cls_prototype in class_prototypes.items()
        }

        proto_to_proto_similarities: torch.Tensor = self.compute_prototypes_similarities(label_prototypes)

        self.label_similarity_dict = self.matrix_to_similarity_dict(proto_to_proto_similarities)

        self.max_difficult_step = datamodule.max_difficult_step

    def sample_labels(self):

        current_step = self.datamodule.trainer.global_step

        first_label = np.random.choice(self.stage_labels, size=1)
        sampled_labels = [first_label.tolist()[0]]

        for i in range(self.episode_hparams.num_classes_per_episode - 1):

            remaining_labels = set(self.stage_labels).difference(sampled_labels)
            remaining_labels_array = np.array(sorted(list(remaining_labels)))

            label_probabilities = self.get_label_probabilities(sampled_labels, remaining_labels, t=current_step)

            label = np.random.choice(remaining_labels_array, size=1, p=label_probabilities)

            sampled_labels.append(label[0])

        return sorted(sampled_labels)

    def compute_prototypes_similarities(self, label_prototypes):
        labels_prototypes = sorted([(k, v) for k, v in label_prototypes.items()], key=lambda t: t[0])
        prototypes = torch.stack([prototype for key, prototype in labels_prototypes])

        interleaved_repeated_prototypes = prototypes.repeat_interleave(len(self.stage_labels), dim=0)

        repeated_prototypes = prototypes.repeat((len(self.stage_labels), 1))

        scaling_factor = torch.load(self.scaling_factor_path)
        distances = scaling_factor * squared_l2(interleaved_repeated_prototypes, repeated_prototypes).reshape(
            (len(self.stage_labels), len(self.stage_labels))
        )

        similarities = 1 / (distances + 1)

        pylogger.info(f"Label similarity matrix: {similarities}")

        return similarities

    def matrix_to_similarity_dict(self, label_similarity_matrix):
        label_similarity_dict = {}

        for ind, stage_label in enumerate(self.stage_labels):
            label_similarity_dict[stage_label] = {
                self.stage_labels[value_ind]: value
                for value_ind, value in enumerate(label_similarity_matrix[ind])
                if ind != value_ind
            }

        return label_similarity_dict

    def get_label_probabilities(self, sampled_labels, remaining_labels, t):

        label_weights = {}

        for label in remaining_labels:
            similarities_label_and_sampled = [
                sim.item() for lbl, sim in self.label_similarity_dict[label].items() if lbl in sampled_labels
            ]
            # similarity_with_sampled = np.sum(np.exp(similarities_label_and_sampled))
            similarity_with_sampled = np.sum(similarities_label_and_sampled)

            label_weight = self.weight_label(similarity_with_sampled, num_steps=t)

            label_weights[label] = label_weight

        # norm_factor = sum(math.exp(j) for j in label_weights.values())
        # label_probabilities = {k: math.exp(v) / norm_factor for k, v in label_weights.items()}
        norm_factor = sum(j for j in label_weights.values())
        label_probabilities = {k: v / norm_factor for k, v in label_weights.items()}

        label_probabilities_array = sorted([(k, v) for k, v in label_probabilities.items()], key=lambda tup: tup[0])
        label_probabilities_array = np.array([v for k, v in label_probabilities_array])

        return label_probabilities_array

    def weight_label(self, similarity_with_sampled, num_steps):
        t = min(num_steps / self.max_difficult_step, 1)

        return (1 - t) * (1 - similarity_with_sampled) + t * similarity_with_sampled

    def __getitem__(self, index):
        raise NotImplementedError
