import abc
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx

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
            episode_global_to_local_mapping = batch.get_global_to_local_label_mapping(episode_global_labels)
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

        if self.plot_graphs:
            self.plot_samples(batch.supports, batch, queries_or_supports="supports")
            self.plot_batch(batch.supports)

            self.plot_samples(batch.queries, batch, queries_or_supports="queries")
            self.plot_batch(batch.queries)

        supports_local_labels = self.get_supports_local_labels(batch)

        batch.supports.x, batch.supports.lens, batch.supports.ptr = self.get_aggregating_nodes_features(
            batch, queries_or_supports="supports"
        )

        batch.supports.x, batch.supports.lens = self.get_prototype_node_features(batch)

        label_to_prototype_node = self.get_label_to_prototype_mapping(batch)

        support_artificial_edge_index = self.get_artificial_edges(
            batch, label_to_prototype_node, "supports", supports_local_labels
        )

        batch.supports.edge_index = support_artificial_edge_index

        embedded_supports = self.embed_supports(batch.supports)

        # shape (num_classes_per_episode, hidden_dim)
        class_prototypes = self.get_class_prototypes(embedded_supports, batch, label_to_prototype_node)

        batch.queries.x, batch.queries.lens, batch.queries.ptr = self.get_aggregating_nodes_features(
            batch, queries_or_supports="queries"
        )

        query_artificial_edge_index = self.get_artificial_edges(
            batch, label_to_prototype_node, "queries", supports_local_labels
        )

        batch.queries.edge_index = query_artificial_edge_index

        embedded_queries = self.embed_queries(batch.queries)

        query_aggregators = self.get_query_aggregators(embedded_queries, batch)

        similarities = self.get_similarities(query_aggregators, class_prototypes, batch)

        if self.plot_graphs:
            self.plot_batch(batch.supports)
            self.plot_batch(batch.queries)

        return {
            "embedded_queries": embedded_queries,
            "class_prototypes": class_prototypes,
            "similarities": similarities,
        }

    def plot_batch(self, samples):

        g = to_networkx(samples, to_undirected=True)
        d = dict(g.degree)
        nx.draw(g, nodelist=d.keys(), node_size=[v * 50 for v in d.values()], with_labels=True)

        plt.show()

    def plot_samples(self, samples, batch: EpisodeBatch, queries_or_supports):
        samples = (
            batch.split_supports_in_episodes()
            if queries_or_supports == "supports"
            else batch.split_queries_in_episodes()
        )

        samples = samples[0]
        samples_one_by_one = samples.to_data_list()

        ncols, nrows = len(samples_one_by_one), 1
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

        for sample, ax in zip(samples_one_by_one, axs):
            g = to_networkx(sample, to_undirected=True)
            nx.draw(g, ax=ax, with_labels=True)

        plt.show()

    def get_aggregating_nodes_features(self, batch: EpisodeBatch, queries_or_supports: str):
        """
        initialize the features for the artificial aggregating nodes
        :param batch:
        :param queries_or_supports:
        :return: features to add, updated lens and updated cumsums
        """

        samples_by_episode = (
            batch.split_supports_in_episodes()
            if queries_or_supports == "supports"
            else batch.split_queries_in_episodes()
        )

        feature_dim = batch.feature_dim

        new_samples_x = []
        new_lens = []

        for episode_samples in samples_by_episode:

            lens = get_lens_from_batch_assignment(episode_samples.batch)

            samples_x_single = episode_samples.x.split(tuple(lens))

            new_samples_x_single = []
            for sample_x in samples_x_single:
                new_node_feature = torch.ones((feature_dim)).unsqueeze(0).type_as(sample_x)
                sample_x = torch.cat((sample_x, new_node_feature), dim=0)

                new_samples_x_single.append(sample_x)

            new_episode_samples_x = torch.cat(new_samples_x_single, dim=0)

            num_samples_per_episode = (
                batch.num_supports_per_episode if queries_or_supports == "supports" else batch.num_queries_per_episode
            )
            assert new_episode_samples_x.shape[0] == episode_samples.num_nodes + num_samples_per_episode

            new_samples_x.append(new_episode_samples_x)

            new_episode_lens = lens + 1
            new_lens.append(new_episode_lens)

        new_samples_x = torch.cat(new_samples_x, dim=0)

        num_nodes = batch.supports.num_nodes if queries_or_supports == "supports" else batch.queries.num_nodes
        assert new_samples_x.shape[0] == num_nodes + num_samples_per_episode * batch.num_episodes

        new_lens = torch.cat(new_lens, dim=0)
        new_ptr = torch.cumsum(new_lens, dim=0)

        return new_samples_x, new_lens, new_ptr

    def get_prototype_node_features(self, batch):
        """
        initialize the features for the artificial prototype nodes
        :return: features containing both node features and added ones, updated lens
        """

        # list of B batches, each collating NxK support graphs
        supports_x_by_episode = batch.split_support_features_in_episodes()
        lens_by_episode = batch.supports.lens.split(tuple([batch.num_supports_per_episode] * batch.num_episodes))

        feature_dim = batch.feature_dim
        new_x = []

        for episode_ind, episode_supports_x in enumerate(supports_x_by_episode):

            # one prototype node per class, initialized with ones
            prototype_nodes_features = torch.ones((batch.episode_hparams.num_classes_per_episode, feature_dim)).type_as(
                episode_supports_x
            )

            new_episode_x = torch.cat((episode_supports_x, prototype_nodes_features), dim=0)
            new_x.append(new_episode_x)

            assert new_episode_x.shape[0] == episode_supports_x.shape[0] + batch.episode_hparams.num_classes_per_episode

            episode_lens = lens_by_episode[episode_ind]
            # the prototype nodes are counted in the nodes of the last sample of the episode
            episode_lens[-1] = episode_lens[-1] + batch.episode_hparams.num_classes_per_episode

        new_x = torch.cat(new_x, dim=0)

        new_lens = torch.cat(lens_by_episode, dim=0)

        return new_x, new_lens

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

    def get_artificial_edges(self, batch, label_to_prototype_mapping, supports_or_queries: str, supports_local_labels):

        ref_tensor = batch.supports.edge_index

        samples_by_episode = (
            batch.split_supports_in_episodes()
            if supports_or_queries == "supports"
            else batch.split_queries_in_episodes()
        )

        all_new_edge_indices = []
        for episode_ind, episode_samples in enumerate(samples_by_episode):

            artificial_edges = []
            lens_edges = episode_samples.num_sample_edges

            increment = torch.cat(
                [torch.tensor([i] * lens_edges[i]).type_as(ref_tensor) for i in range(len(lens_edges))],
                dim=0,
            )

            incremented_edge_index = episode_samples.edge_index + increment
            lens = get_lens_from_batch_assignment(episode_samples.batch)

            num_samples_per_episode = (
                batch.num_supports_per_episode if supports_or_queries == "supports" else batch.num_queries_per_episode
            )

            for ind in range(num_samples_per_episode):

                aggregator_edges = self.get_aggregator_edges(lens, episode_samples.ptr, ind)

                if supports_or_queries == "supports":
                    pooling_to_prototype_edges = self.get_aggregator_to_prototype_edge(
                        episode_samples.y[ind], episode_samples.ptr, ind, label_to_prototype_mapping[episode_ind]
                    )

                    aggregator_edges.extend(pooling_to_prototype_edges)

                artificial_edges.extend(aggregator_edges)

            artificial_edge_index = torch.tensor(artificial_edges).squeeze().transpose(1, 0).type_as(ref_tensor)
            new_edge_index = torch.cat((incremented_edge_index, artificial_edge_index), dim=1)
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
            edges.append([u, v])
            edges.append([v, u])

        return edges

    def get_aggregator_to_prototype_edge(self, label, cumsums, ind, label_to_prototype_node):
        label_prototype_node = label_to_prototype_node[label.item()]

        aggregator_node_index = cumsums[ind + 1] + ind
        u, v = aggregator_node_index.item(), label_prototype_node
        pooling_to_prototype_edges = [[u, v], [v, u]]

        return pooling_to_prototype_edges

    def get_class_prototypes(self, embedded_supports, batch, label_to_prototype_node):

        embedded_supports_one_by_one = embedded_supports.split(tuple(batch.supports.lens))

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
            {key: episode_supports[value] for key, value in label_to_prototype_node[episode_ind].items()}
            for episode_ind, episode_supports in enumerate(embedded_supports_by_episodes)
        ]

    def get_query_aggregators(self, embedded_queries, batch):

        aggregator_indices = batch.queries.lens - 1
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
