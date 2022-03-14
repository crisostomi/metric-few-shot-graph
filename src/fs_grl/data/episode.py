import copy
import itertools
from dataclasses import dataclass
from typing import Dict, List, Union

import networkx as nx
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx

from fs_grl.data.utils import flatten


@dataclass
class EpisodeHParams:
    """
    num_supports_per_class: K, how many labeled samples there are for each class when doing few-shot
    num_queries_per_class: Q, how many samples must be predicted for each class when doing few-shot
    num_classes_per_episode: N, how many classes are considered in one few-shot episode
    """

    num_classes_per_episode: int
    num_supports_per_class: int
    num_queries_per_class: int

    def as_dict(self):
        return {
            "num_classes_per_episode": self.num_classes_per_episode,
            "num_supports_per_class": self.num_supports_per_class,
            "num_queries_per_class": self.num_queries_per_class,
        }


class Episode:
    def __init__(
        self,
        supports: Union[List[Data], Batch],
        queries: Union[List[Data], Batch],
        global_labels: List[int],
        episode_hparams: EpisodeHParams,
    ):
        """
        N classes, K samples each, Q queries each

        :param supports: shape (N*K), contains K support samples for each class
        :param queries: shape (N*Q), contains Q queries for each class
        :param global_labels: subset of the stage labels sampled for the episode
        :param episode_hparams: N, K and Q
        """
        self.supports = supports
        self.queries = queries
        self.global_labels = global_labels

        self.episode_hparams = episode_hparams
        self.num_queries_per_episode = (
            self.episode_hparams.num_queries_per_class * self.episode_hparams.num_classes_per_episode
        )
        self.num_supports_per_episode = (
            self.episode_hparams.num_supports_per_class * self.episode_hparams.num_classes_per_episode
        )


class EpisodeBatch(Episode):
    def __init__(
        self,
        supports: Batch,
        queries: Batch,
        global_labels: torch.Tensor,
        episode_hparams: EpisodeHParams,
        num_episodes: int,
        cosine_targets: torch.Tensor,
        local_labels: torch.Tensor,
        label_to_prototype_mapping: Dict,
    ):
        """

        :param supports: supports for all the episodes in the batch
        :param queries: queries for all the episodes in the batch
        :param global_labels: tensor containing for each episode the considered global labels
        :param episode_hparams: N, K, Q
        :param num_episodes: how many episodes per batch
        :param cosine_targets:
        :param local_labels: labels remapped to 0,..,N-1 which are local to the single episodes
                             i.e. they are not consistent through the batch
        """
        super().__init__(
            supports=supports, queries=queries, global_labels=global_labels, episode_hparams=episode_hparams
        )

        self.num_episodes = num_episodes
        self.cosine_targets = cosine_targets
        self.local_labels = local_labels
        self.label_to_prototype_mapping = label_to_prototype_mapping

    @classmethod
    def from_episode_list(
        cls, episode_list: List[Episode], episode_hparams: EpisodeHParams, add_prototypes=True, plot_graphs=False
    ) -> "EpisodeBatch":

        episode_list = [copy.deepcopy(episode) for episode in episode_list]

        if add_prototypes:
            label_to_prototype_mappings = []
            all_prototype_edges = []

            cumsum = 0
            for episode in episode_list:

                last_support: Data = episode.supports[-1]

                cls.add_prototype_features(last_support, episode_hparams)
                old_cumsum = cumsum

                cumsum += sum([support.num_nodes for support in episode.supports])

                label_to_prototype_mapping = cls.get_label_to_prototype_mapping(episode, episode_hparams, cumsum)
                prototype_edges = cls.add_prototype_edges(
                    episode, label_to_prototype_mapping, episode_hparams, old_cumsum
                )

                label_to_prototype_mappings.append(label_to_prototype_mapping)
                all_prototype_edges.append(prototype_edges)

        # N * K * batch_size
        supports: List[Data] = flatten([episode.supports for episode in episode_list])

        # N * Q * batch_size
        queries: List[Data] = flatten([episode.queries for episode in episode_list])
        # N * batch_size
        global_labels: List[int] = flatten([episode.global_labels for episode in episode_list])

        supports_batch: Batch = Batch.from_data_list(supports)

        if add_prototypes:
            # TODO: check if it works with batching
            supports_by_episodes = [Batch.from_data_list(episode.supports) for episode in episode_list]

            batch_edge_index = []
            cumsum = 0
            for episode_ind, episode_supports in enumerate(supports_by_episodes):
                episode_edges = episode_supports.edge_index
                prototype_edges = all_prototype_edges[episode_ind]

                episode_edges += cumsum
                episode_edges = torch.cat((episode_edges, prototype_edges), dim=1)

                cumsum += episode_supports.num_nodes

                batch_edge_index.append(episode_edges)

            batch_edge_index = torch.cat(batch_edge_index, dim=1)
            supports_batch.edge_index = batch_edge_index
            supports_batch.num_edges += episode_hparams.num_classes_per_episode

        queries_batch: Batch = Batch.from_data_list(queries)
        global_labels_batch = torch.tensor(global_labels)

        num_episodes = len(episode_list)

        # shape (B*N*Q*N)
        cosine_targets = cls.get_cosine_targets(episode_list)

        # shape (N*Q*B, N)
        local_labels = cosine_targets.reshape(-1, episode_hparams.num_classes_per_episode)
        local_labels = local_labels.argmax(dim=-1)

        episode_batch = cls(
            supports=supports_batch,
            queries=queries_batch,
            global_labels=global_labels_batch,
            episode_hparams=episode_hparams,
            num_episodes=num_episodes,
            cosine_targets=cosine_targets,
            local_labels=local_labels,
            label_to_prototype_mapping=label_to_prototype_mappings,
        )

        if plot_graphs:
            episode_batch.plot_batch(queries_batch, "queries")
            episode_batch.plot_batch(supports_batch, "supports")

        return episode_batch

    def plot_batch(self, samples, supports_or_queries):
        g = to_networkx(samples, to_undirected=False)
        pos = nx.nx_agraph.graphviz_layout(g, prog="twopi", args="")

        plt.figure(figsize=(16, 16))

        aggregator_nodes = [node.item() for node in self.get_aggregator_indices(supports_or_queries)]
        all_nodes = set(range(samples.num_nodes))
        normal_nodes = all_nodes.difference(aggregator_nodes)

        normal_nodes_labels = {k: k for k, v in pos.items() if k in normal_nodes}
        aggregator_nodes_labels = {k: k for k, v in pos.items() if k in aggregator_nodes}

        nx.draw_networkx_nodes(g, pos, nodelist=normal_nodes, node_color="tab:grey", node_size=40, alpha=0.5)
        nx.draw_networkx_labels(g, pos, labels=normal_nodes_labels, alpha=0.5)

        nx.draw_networkx_nodes(g, pos, nodelist=aggregator_nodes, node_color="tab:blue", node_size=80, alpha=0.5)
        nx.draw_networkx_labels(g, pos, labels=aggregator_nodes_labels, alpha=0.5)

        if supports_or_queries == "supports":
            prototype_nodes = {v for mapping in self.label_to_prototype_mapping for k, v in mapping.items()}

        normal_edges = []
        aggregator_edges = []
        prototype_edges = []
        for edge in g.edges:
            u, v = edge
            if v in aggregator_nodes:
                aggregator_edges.append(edge)
            elif supports_or_queries == "supports" and v in prototype_nodes:
                prototype_edges.append(edge)
            else:
                normal_edges.append(edge)

        nx.draw_networkx_edges(g, pos, edgelist=normal_edges, edge_color="tab:grey")
        nx.draw_networkx_edges(g, pos, edgelist=aggregator_edges, edge_color="tab:blue", style="dashed")

        if supports_or_queries == "supports":
            prototype_nodes_labels = {k: k for k, v in pos.items() if k in prototype_nodes}

            nx.draw_networkx_nodes(g, pos, nodelist=prototype_nodes, node_color="#FAC60E", node_size=160, alpha=0.5)
            nx.draw_networkx_labels(g, pos, labels=prototype_nodes_labels, alpha=0.5)
            nx.draw_networkx_edges(g, pos, edgelist=prototype_edges, edge_color="#FAC60E", style="dashed")

        plt.axis("equal")

        plt.show()

    def get_aggregator_indices(self, queries_or_supports):
        if queries_or_supports == "queries":
            return self.queries.ptr[1:] - 1
        else:

            ptr = self.supports.ptr[1:]
            ptr_by_episode = ptr.split(tuple([self.num_supports_per_episode] * self.num_episodes))
            aggregator_indices = [episode_ptr - 1 for episode_ptr in ptr_by_episode]
            for episode_aggregators in aggregator_indices:
                episode_aggregators[-1] -= self.episode_hparams.num_classes_per_episode
            aggregator_indices = torch.cat(aggregator_indices, dim=0)
            return aggregator_indices

    #
    # def to_episode_list(self):
    #
    #     episodes = []
    #
    #     supports_flattened = self.supports.to_data_list()
    #     queries_flattened = self.queries.to_data_list()
    #
    #     supports_by_episodes = [
    #         supports_flattened[
    #             i * self.num_supports_per_episode : i * self.num_supports_per_episode + self.num_supports_per_episode
    #         ]
    #         for i in range(self.num_episodes)
    #     ]
    #
    #     queries_by_episodes = [
    #         queries_flattened[
    #             i * self.num_queries_per_episode : i * self.num_queries_per_episode + self.num_queries_per_episode
    #         ]
    #         for i in range(self.num_episodes)

    #     ]

    # TODO: finish

    @classmethod
    def get_global_to_local_label_mapping(cls, global_labels):
        # assuming we are in an episode
        return {glob.item(): local for local, glob in enumerate(sorted(global_labels))}

    @classmethod
    def get_cosine_targets(cls, episode_list: List[Episode]) -> torch.Tensor:
        """
        :param episode_list: list of episodes in the batch
        :return: tensor ~(B*(N*Q)*N) where for each episode in [1, .., B] there are all the
                 target similarities between the N*Q queries and the N considered global labels
                 Query q in [1, .., (N*Q)] and label l in [1, .., N] will have sim(q, l) = 1 if
                 query q has label l, else -1
        """
        cosine_targets = []
        for episode in episode_list:

            # shape (N*Q*N)
            episode_cosine_targets = []

            for query, label in itertools.product(episode.queries, episode.global_labels):
                query_label_similarity = (query.y.item() == label) * 2 - 1
                episode_cosine_targets.append(query_label_similarity)

            cosine_targets.append(torch.tensor(episode_cosine_targets, dtype=torch.long))

        cosine_targets = torch.cat(cosine_targets, dim=-1)

        return cosine_targets

    def to(self, device):
        self.supports = self.supports.to(device)
        self.queries = self.queries.to(device)
        self.cosine_targets = self.cosine_targets.to(device)
        self.local_labels = self.local_labels.to(device)
        self.global_labels = self.global_labels.to(device)

    def pin_memory(self):
        for key, attr in self.__dict__.items():
            if attr is not None and hasattr(attr, "pin_memory"):
                attr.pin_memory()

        return self

    def split_supports_in_episodes(self) -> list:
        supports_data_list = self.supports.to_data_list()

        supports_by_episodes = [
            supports_data_list[
                i * self.num_supports_per_episode : i * self.num_supports_per_episode + self.num_supports_per_episode
            ]
            for i in range(self.num_episodes)
        ]

        supports_by_episodes = [Batch.from_data_list(episode_supports) for episode_supports in supports_by_episodes]

        assert self.supports.num_nodes == sum([supports.num_nodes for supports in supports_by_episodes])
        return supports_by_episodes

    def split_queries_in_episodes(self):
        queries_data_list = self.queries.to_data_list()

        queries_by_episodes = [
            queries_data_list[
                i * self.num_queries_per_episode : i * self.num_queries_per_episode + self.num_queries_per_episode
            ]
            for i in range(self.num_episodes)
        ]

        queries_by_episodes = [Batch.from_data_list(episode_queries) for episode_queries in queries_by_episodes]

        assert self.queries.num_nodes == sum([queries.num_nodes for queries in queries_by_episodes])
        return queries_by_episodes

    def split_queries_features_in_episodes(self) -> tuple:
        return self.queries.x.split(tuple([self.num_queries_per_episode] * self.num_episodes))

    def split_support_features_in_episodes(self) -> List:
        supports_x_one_by_one = self.supports.x.split(tuple(self.supports.lens))

        supports_x_by_episodes = [
            torch.cat(
                supports_x_one_by_one[
                    i * self.num_supports_per_episode : i * self.num_supports_per_episode
                    + self.num_supports_per_episode
                ],
                dim=0,
            )
            for i in range(self.num_episodes)
        ]
        return supports_x_by_episodes

    def split_supports_labels_in_episodes(self) -> tuple:
        return self.supports.y.split(tuple([self.num_supports_per_episode] * self.num_episodes))

    def split_queries_labels_in_episodes(self) -> tuple:
        return self.queries.y.split(tuple([self.num_queries_per_episode] * self.num_episodes))

    def split_supports_ptr_in_episodes(self) -> tuple:
        return self.supports.ptr.split(tuple([self.num_supports_per_episode] * self.num_episodes))

    @property
    def feature_dim(self):
        assert self.supports.x.shape[-1] == self.queries.x.shape[-1]
        return self.supports.x.shape[-1]

    @classmethod
    def add_prototype_features(cls, last_support, episode_hparams):
        last_support.num_nodes += episode_hparams.num_classes_per_episode
        feature_dim = last_support.x.shape[-1]
        prototype_features = torch.ones((episode_hparams.num_classes_per_episode, feature_dim)).type_as(last_support.x)
        last_support.x = torch.cat((last_support.x, prototype_features), dim=0)

    @classmethod
    def get_label_to_prototype_mapping(cls, episode, episode_hparams, cumsum):

        supports = episode.supports
        prototype_index = cumsum
        episode_global_labels = torch.unique(torch.tensor([support.y for support in supports]))
        sorted_episode_global_labels = torch.sort(episode_global_labels).values

        episode_label_to_prot = {
            global_label.item(): prototype_index - (ind + 1)
            for ind, global_label in enumerate(sorted_episode_global_labels)
        }
        return episode_label_to_prot

    @classmethod
    def add_prototype_edges(cls, episode, label_to_prototype_mapping, episode_hparams, cumsum):
        supports = episode.supports
        pooling_to_prototype_edges = []

        episode_cumsum = 0
        for ind, support in enumerate(supports):
            label_prototype_node = label_to_prototype_mapping[support.y.item()]

            # TODO: check correctness
            episode_cumsum += (
                support.num_nodes - 1
                if ind != (len(supports) - 1)
                else support.num_nodes - episode_hparams.num_classes_per_episode - 1
            )
            aggregator_node_index = episode_cumsum + ind + cumsum
            u, v = aggregator_node_index, label_prototype_node
            pooling_to_prototype_edge = [u, v]
            pooling_to_prototype_edges.append(pooling_to_prototype_edge)

        return torch.tensor(pooling_to_prototype_edges).transpose(1, 0)

    def plot_samples(self, batch: "EpisodeBatch", queries_or_supports):
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
            g = to_networkx(sample, to_undirected=False)
            nx.draw(g, ax=ax, with_labels=True)

        plt.show()
