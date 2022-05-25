import json
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import numpy as np
import ot
import scipy
import torch
from networkx import normalized_laplacian_matrix
from omegaconf import DictConfig
from sklearn.cluster import AgglomerativeClustering

from fs_grl.data.datamodule import GraphTransferDataModule
from fs_grl.data.episode import EpisodeHParams

pylogger = logging.getLogger(__name__)


class GSMDataModule(GraphTransferDataModule):
    def __init__(
        self,
        dataset_name,
        feature_params: Dict,
        data_dir,
        classes_split_path: Optional[str],
        query_support_split_path,
        separated_query_support: bool,
        support_ratio,
        test_episode_hparams: EpisodeHParams,
        num_train_episodes,
        num_val_episodes,
        num_test_episodes,
        train_ratio,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        spectral_prototypes_path: str,
        num_clusters: Dict,
        **kwargs,
    ):
        super().__init__(
            dataset_name=dataset_name,
            feature_params=feature_params,
            data_dir=data_dir,
            classes_split_path=classes_split_path,
            query_support_split_path=query_support_split_path,
            separated_query_support=separated_query_support,
            support_ratio=support_ratio,
            train_ratio=train_ratio,
            num_train_episodes=num_train_episodes,
            num_val_episodes=num_val_episodes,
            num_test_episodes=num_test_episodes,
            test_episode_hparams=test_episode_hparams,
            num_workers=num_workers,
            batch_size=batch_size,
            gpus=gpus,
            add_artificial_nodes=False,
            artificial_node_features=None,
        )

        self.num_clusters = num_clusters

        label_to_prototype_index_dict = self.create_or_load_spectral_prototypes(spectral_prototypes_path)
        self.label_to_prototype_index_dict = OrderedDict(
            sorted(label_to_prototype_index_dict.items(), key=lambda t: t[0])
        )

        self.label_to_prototype_dict = {
            k: self.graph_list[ind] for k, ind in self.label_to_prototype_index_dict.items()
        }

        self.cls_to_supercls_mapping = self.get_super_classes()

        for data in self.data_list:
            data.super_class = torch.tensor(self.cls_to_supercls_mapping[data.y.item()])

    def create_or_load_spectral_prototypes(self, spectral_prototypes_path):
        """

        :param spectral_prototypes_path:
        :return:
        """

        if not os.path.exists(spectral_prototypes_path):
            label_to_prototype_index_dict = self.create_spectral_prototypes()

            with open(spectral_prototypes_path, "w+") as f:
                json.dump(label_to_prototype_index_dict, f)
        else:
            with open(spectral_prototypes_path, "r") as f:
                label_to_prototype_index_dict = json.load(f)
                label_to_prototype_index_dict = {int(k): v for k, v in label_to_prototype_index_dict.items()}

        return label_to_prototype_index_dict

    def create_spectral_prototypes(self):
        """
        Create prototypes for given dataset according to GSM
        https://arxiv.org/abs/2002.12815

        :param graph_list_by_base_label:
        :return:
        """

        graph_eigenvals_by_label = {label: [] for label in self.graph_list_by_label.keys()}

        for label, label_graph_list in self.graph_list_by_label.items():

            for G in label_graph_list:

                N = normalized_laplacian_matrix(G).todense()

                eigvals = scipy.linalg.eigvals(N)
                eigvals = eigvals.real.round(decimals=5)

                graph_eigenvals_by_label[label].append(eigvals)

        label_to_prototype_index_dict = {}

        for label, label_graphs_eigenvals in graph_eigenvals_by_label.items():

            all_distances = self.compute_spectral_distances(label_graphs_eigenvals)

            proto_index = np.argmin(np.sum(all_distances, axis=1))

            proto_dataset_index = self.graph_list_by_label[label][proto_index].graph["dataset_index"]
            label_to_prototype_index_dict[label] = proto_dataset_index

        return label_to_prototype_index_dict

    def compute_spectral_distances(self, graph_eigenvals: List[np.array]) -> np.array:
        """

        :param graph_eigenvals:
        :return:
        """

        distance_matrix = []

        for graph_i_eigenvals in graph_eigenvals:
            current_dist = []

            for graph_j_eigenvals in graph_eigenvals:
                a, b = graph_i_eigenvals, graph_j_eigenvals

                cost = ot.utils.dist(np.reshape(a, (a.shape[0], 1)), np.reshape(b, (b.shape[0], 1)))

                # Uniform distribution has been assumed over the spectra for faster implementation. One can first use density estimation
                # to approximate the distribution which can provide better results.
                loss = ot.emd2([], [], cost)

                current_dist.append(loss)

            distance_matrix.append(current_dist)

        all_dist = np.array(distance_matrix)
        return all_dist

    def get_super_classes(self):
        """

        :return:
        """
        # list containing the first 5 eigenvalues for each prototype

        prototypes_eigenvals = self.get_prototypes_eigenvals()

        kernel_matrix = self.get_similarity_matrix(prototypes_eigenvals)

        # This is used as an approximation with the Lloyd's variant proposed in the paper. Well intergrated with the scikit-learn
        # library, its assures better implementation and was thus used in the final version.
        clustering_super_class = AgglomerativeClustering(
            n_clusters=self.num_clusters["train"], affinity="precomputed", linkage="complete"
        ).fit(kernel_matrix)

        super_class_dict = {}
        cls_to_supercls_mapping = {}

        classes = list(self.label_to_prototype_dict.keys())
        classes.sort()

        super_class_labels = list(clustering_super_class.labels_)

        for supercls, cls in zip(super_class_labels, classes):

            if supercls not in super_class_dict.keys():
                super_class_dict[supercls] = []

            super_class_dict[supercls].append(cls)
            cls_to_supercls_mapping[cls] = supercls

        return cls_to_supercls_mapping

    def get_prototypes_eigenvals(self):
        prototypes_eigenvals = []

        for cls, cls_spec_prototype in self.label_to_prototype_dict.items():

            L = normalized_laplacian_matrix(cls_spec_prototype).todense()
            eigvals = scipy.linalg.eigvals(L)
            eigvals = eigvals.real.round(decimals=5)

            assert type(eigvals) != int

            prototypes_eigenvals.append(eigvals)

        return prototypes_eigenvals

    def get_similarity_matrix(self, prototypes_eigenvals):
        """

        :param prototypes_eigenvals:
        :return:
        """
        all_dist = []

        for i in range(len(prototypes_eigenvals)):
            current_dist = []

            for j in range(len(prototypes_eigenvals)):
                a = prototypes_eigenvals[i]
                b = prototypes_eigenvals[j]
                cost = ot.utils.dist(np.reshape(a, (a.shape[0], 1)), np.reshape(b, (b.shape[0], 1)))
                loss = ot.emd2([], [], cost)
                current_dist.append(loss)

            all_dist.append(current_dist)

        kernel_matrix = np.array(all_dist)
        return kernel_matrix
