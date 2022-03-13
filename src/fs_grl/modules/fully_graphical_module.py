import abc
from typing import Dict, List

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch_geometric.data import Batch

from fs_grl.data.episode import EpisodeBatch
from fs_grl.data.utils import get_lens_from_batch_assignment
from fs_grl.modules.losses.margin import MarginLoss
from fs_grl.modules.similarities.cosine import cosine


class FullyGraphicalModule(nn.Module, abc.ABC):
    def __init__(self, cfg, feature_dim, num_classes, margin, plot_graphs, **kwargs):
        super().__init__()
        self.cfg = cfg

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.plot_graphs = plot_graphs

        self.embedder = instantiate(
            self.cfg.embedder,
            feature_dim=self.feature_dim,
        )

        self.compute_global_prototypes = False
        self.loss_func = MarginLoss(margin=margin, reduction="mean")

    def embed_supports(self, supports: Batch):
        """
        :param supports: Batch containing BxNxK support graphs as a single large graph
        :return: embedded supports ~ ((B*N*K)xE), each graph embedded as a point in R^{E}
        """
        return self._embed(supports)

    def embed_queries(self, queries: Batch):
        """
        :param queries: Batch containing BxNxQ query graphs
        :return: embedded queries ~ (BxNxQxE), each graph embedded as a point in R^{E}
        """
        return self._embed(queries)

    def _embed(self, batch: Batch):
        """
        Embeds a batch of graphs given as a single large graph

        :param batch: Batch containing graphs to embed
        :return: embedded graphs, each graph embedded as a point in R^{E}
        """

        embedded_batch = self.embedder(batch)
        return embedded_batch

    def get_similarities(self, embedded_queries, class_prototypes, batch):
        batch_queries_prototypes = self.align_queries_prototypes(batch, embedded_queries, class_prototypes)
        batch_queries, batch_prototypes = batch_queries_prototypes["queries"], batch_queries_prototypes["prototypes"]

        similarities = cosine(batch_queries, batch_prototypes)

        return similarities

    def get_supports_local_labels(self, batch: EpisodeBatch):
        global_labels_by_episode = batch.supports.y.split(tuple([batch.num_supports_per_episode] * batch.num_episodes))

        local_labels_by_episode = []
        for episode_global_labels in global_labels_by_episode:
            episode_global_to_local_mapping = batch.get_global_to_local_label_mapping(
                torch.unique(episode_global_labels)
            )
            episode_local_labels = torch.tensor(
                [episode_global_to_local_mapping[y.item()] for y in episode_global_labels]
            ).type_as(batch.supports.edge_index)

            local_labels_by_episode.append(episode_local_labels)

        local_labels_by_episode = torch.cat(local_labels_by_episode, dim=0)

        return local_labels_by_episode

    def forward(self, batch: EpisodeBatch):
        """
        :param batch:
        :return:
        """

        embedded_supports = self.embed_supports(batch.supports)

        # shape (num_classes_per_episode, hidden_dim)
        class_prototypes = self.get_class_prototypes(embedded_supports, batch, batch.label_to_prototype_mapping)

        embedded_queries = self.embed_queries(batch.queries)

        query_aggregators = self.get_query_aggregators(embedded_queries, batch)

        similarities = self.get_similarities(query_aggregators, class_prototypes, batch)

        return {
            "embedded_queries": embedded_queries,
            "class_prototypes": class_prototypes,
            "similarities": similarities,
        }

    def get_label_to_prototype_mapping(self, batch):
        """
        return a list (num_episodes,) of dictionaries, each dictionary maps a local label
        to the corresponding prototype index
        :param batch:
        :return:
        """

        supports_x_by_episode = batch.split_support_features_in_episodes()

        global_labels_by_episode = batch.supports.y.split(tuple([batch.num_supports_per_episode] * batch.num_episodes))
        label_to_prototype_mappings = []

        for episode_supports_x, episode_global_labels in zip(supports_x_by_episode, global_labels_by_episode):
            episode_global_labels = torch.unique(episode_global_labels)
            sorted_episode_global_labels = torch.sort(episode_global_labels).values

            episode_label_to_prot = {
                global_label.item(): len(episode_supports_x) - (ind + 1)
                for ind, global_label in enumerate(sorted_episode_global_labels)
            }

            label_to_prototype_mappings.append(episode_label_to_prot)

        return label_to_prototype_mappings

    def compute_loss(self, model_out, batch: EpisodeBatch, **kwargs):
        similarities = model_out["similarities"]

        return self.loss_func(similarities, batch.cosine_targets)

    def get_artificial_edges(self, batch, label_to_prototype_mapping):
        # TODO: delete

        ref_tensor = batch.supports.edge_index

        supports_by_episode = batch.split_supports_in_episodes()

        all_new_edge_indices = []

        for episode_ind, episode_samples in enumerate(supports_by_episode):

            artificial_edges = []
            # lens_edges = episode_samples.num_sample_edges

            # TODO: fix
            # increment = torch.cat(
            #     [torch.tensor([i] * lens_edges[i]).type_as(ref_tensor) for i in range(len(lens_edges))],
            #     dim=0,
            # )
            # incremented_edge_index = episode_samples.edge_index + increment

            num_supports_per_episode = batch.num_supports_per_episode

            for ind in range(num_supports_per_episode):

                pooling_to_prototype_edges = self.get_aggregator_to_prototype_edge(
                    episode_samples.y[ind], episode_samples.ptr, ind, label_to_prototype_mapping[episode_ind]
                )

                artificial_edges.extend(pooling_to_prototype_edges)

            artificial_edge_index = torch.tensor(artificial_edges).squeeze().transpose(1, 0).type_as(ref_tensor)
            new_edge_index = torch.cat((episode_samples.edge_index, artificial_edge_index), dim=1)
            all_new_edge_indices.append(new_edge_index)

        result = torch.cat(all_new_edge_indices, dim=1)

        return result

    def get_aggregator_edges(self, lens, cumsums, ind):

        lens = torch.cat((torch.tensor([0]).type_as(lens), lens), dim=0)

        aggregator_node_index = cumsums[ind + 1] + ind
        starting_index = cumsums[ind] + (ind)
        num_nodes = lens[ind + 1]

        edges = []
        for node in torch.arange(starting_index, starting_index + num_nodes):
            u, v = aggregator_node_index.item(), node.item()
            # edges.append([u, v])
            edges.append([v, u])

        return edges

    def get_aggregator_to_prototype_edge(self, label, cumsums, ind, label_to_prototype_node):
        label_prototype_node = label_to_prototype_node[label.item()]

        aggregator_node_index = cumsums[ind + 1] - 1
        u, v = aggregator_node_index.item(), label_prototype_node
        pooling_to_prototype_edges = [[u, v]]

        return pooling_to_prototype_edges

    def get_class_prototypes(self, embedded_supports, batch, label_to_prototype_mapping):

        # TODO: should this be updated?
        lens = get_lens_from_batch_assignment(batch.supports.batch)
        embedded_supports_one_by_one = embedded_supports.split(tuple(lens))

        embedded_supports_by_episodes = [
            torch.cat(
                embedded_supports_one_by_one[
                    i * batch.num_supports_per_episode : i * batch.num_supports_per_episode
                    + batch.num_supports_per_episode
                ],
                dim=0,
            )
            for i in range(batch.num_episodes)
        ]

        return [
            {key: episode_supports[value] for key, value in label_to_prototype_mapping[episode_ind].items()}
            for episode_ind, episode_supports in enumerate(embedded_supports_by_episodes)
        ]

    def get_query_aggregators(self, embedded_queries, batch):

        aggregator_indices = batch.get_aggregator_indices("queries")
        embedded_aggregators = embedded_queries[aggregator_indices]

        return embedded_aggregators

    def align_queries_prototypes(
        self, batch, embedded_queries: torch.Tensor, class_prototypes: List[Dict[int, torch.Tensor]]
    ):
        """

        :param batch:
        :param embedded_queries: shape (num_queries_batch, hidden_dim)
        :param class_prototypes:
        :return:
        """

        num_episodes = batch.num_episodes

        batch_queries = []
        batch_prototypes = []
        embedded_queries_per_episode = embedded_queries.split(tuple([batch.num_queries_per_episode] * num_episodes))

        for episode in range(num_episodes):

            class_prototype_matrix = self.get_prototype_matrix_from_dict(class_prototypes[episode])

            aligned_queries, aligned_prototypes = self.align_queries_prototypes_pairs(
                embedded_queries_per_episode[episode], class_prototype_matrix, batch
            )

            batch_queries.append(aligned_queries)
            batch_prototypes.append(aligned_prototypes)

        return {"queries": torch.cat(batch_queries, dim=0), "prototypes": torch.cat(batch_prototypes, dim=0)}

    def align_queries_prototypes_pairs(self, queries, prototypes_matrix, batch):
        # shape (num_queries_episode, hidden_dim)

        aligned_embedded_queries = queries.repeat_interleave(batch.episode_hparams.num_classes_per_episode, dim=0)

        aligned_prototypes = prototypes_matrix.repeat((batch.num_queries_per_episode, 1))

        return aligned_embedded_queries, aligned_prototypes

    @classmethod
    def get_prototype_matrix_from_dict(cls, class_prototypes):
        sorted_class_prototypes = [(global_class, prototype) for global_class, prototype in class_prototypes.items()]
        sorted_class_prototypes.sort(key=lambda tup: tup[0])
        sorted_class_prototypes_tensors = [tup[1] for tup in sorted_class_prototypes]

        # shape (num_classes_episode, hidden_dim)
        class_prototype_matrix = torch.stack(sorted_class_prototypes_tensors)

        return class_prototype_matrix
