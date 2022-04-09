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
        global_labels: torch.Tensor,
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
        label_to_prototype_mapping: List[Dict] = None,
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
        cls,
        episode_list: List[Episode],
        episode_hparams: EpisodeHParams,
        add_prototype_nodes: bool = False,
        plot_graphs: bool = False,
        artificial_node_features: str = "",
    ) -> "EpisodeBatch":

        label_to_prototype_mappings = None

        if add_prototype_nodes:
            # for each episode, get the label -> prototype node mapping and the aggregator -> prototype edges
            episode_list = [copy.deepcopy(episode) for episode in episode_list]
            all_prototype_edges, label_to_prototype_mappings = cls.handle_prototype_nodes_and_edges(
                episode_list, episode_hparams, artificial_node_features
            )

        # B * N * K
        supports: List[Data] = flatten([episode.supports for episode in episode_list])
        # B * N * Q
        queries: List[Data] = flatten([episode.queries for episode in episode_list])
        # B * N
        global_labels: List[int] = flatten([episode.global_labels for episode in episode_list])

        supports_batch: Batch = Batch.from_data_list(supports)
        queries_batch: Batch = Batch.from_data_list(queries)
        global_labels_batch = torch.tensor(global_labels)

        if add_prototype_nodes:
            cls.update_supports_batch_with_prototype_edges(
                supports_batch, episode_list, all_prototype_edges, episode_hparams
            )

        # shape (B*(N*Q)*N)
        cosine_targets = cls.get_cosine_targets(episode_list)
        # shape (N*(Q*B))
        local_labels = cls.get_local_labels_from_cosine_targets(cosine_targets, episode_hparams)

        episode_batch = cls(
            supports=supports_batch,
            queries=queries_batch,
            global_labels=global_labels_batch,
            episode_hparams=episode_hparams,
            num_episodes=len(episode_list),
            cosine_targets=cosine_targets,
            local_labels=local_labels,
            label_to_prototype_mapping=label_to_prototype_mappings,
        )

        if add_prototype_nodes and plot_graphs:
            episode_batch.plot_batch(queries_batch, "queries")
            episode_batch.plot_batch(supports_batch, "supports")

        return episode_batch

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

            # shape ((N*Q)*N)
            episode_cosine_targets = []

            for query, label in itertools.product(episode.queries, episode.global_labels):
                query_label_similarity = (query.y.item() == label) * 2 - 1
                episode_cosine_targets.append(query_label_similarity)

            cosine_targets.append(torch.tensor(episode_cosine_targets, dtype=torch.long))

        cosine_targets = torch.cat(cosine_targets, dim=-1)

        return cosine_targets

    @classmethod
    def get_local_labels_from_cosine_targets(
        cls, cosine_targets: torch.Tensor, episode_hparams: EpisodeHParams
    ) -> torch.Tensor:
        """

        :param cosine_targets: tensor ~ (B*(N*Q)*N), where for each query we have 1 for the corresponding class
                                and -1 for the others
        :param episode_hparams: N, K, Q

        :return local_labels: tensor ~ (B*N*Q) where for each query we have the corresponding class index
        """
        cosine_targets_by_label = cosine_targets.reshape(-1, episode_hparams.num_classes_per_episode)
        local_labels = cosine_targets_by_label.argmax(dim=-1)

        return local_labels

    @classmethod
    def handle_prototype_nodes_and_edges(
        cls, episode_list: List[Episode], episode_hparams: EpisodeHParams, artificial_node_features
    ):
        """

        :param episode_list: List (num_episodes) containing the Episodes
        :param episode_hparams: N, K, Q

        :return all_prototype_edges: List[Tensor] ~ (num_episodes, N*K) containing the prototype edges for each episode
        :return label_to_prototype_mappings: List[Dict] ~ (num_episodes,) containing for each episode the mapping
                                             (label --> prototype node index) for each global label
        """

        all_prototype_edges = []
        label_to_prototype_mappings = []

        # at episode i, contains the total number of nodes of all the supports of the previous episodes
        cumsum = 0
        for episode in episode_list:

            # node features for the prototype nodes are added to the last support
            cls.add_prototype_features(episode.supports, episode_hparams, artificial_node_features)

            old_cumsum = cumsum

            # total number of nodes of the support graphs
            cumsum += sum([support.num_nodes for support in episode.supports])

            # map each global label to a prototype node index
            label_to_prototype_mapping = cls.get_label_to_prototype_mapping(episode, cumsum)

            prototype_edges = cls.get_aggregator_prototype_edges(
                episode, label_to_prototype_mapping, episode_hparams, old_cumsum
            )

            label_to_prototype_mappings.append(label_to_prototype_mapping)

            all_prototype_edges.append(prototype_edges)

        return all_prototype_edges, label_to_prototype_mappings

    @classmethod
    def add_prototype_features(cls, supports: List[Data], episode_hparams: EpisodeHParams, artificial_node_features):
        """
        Add the node features for the prototype nodes. Currently, these are just ones.
        They are concatenated to the feature matrix of the last support graph.
        last_support is updated in place.

        :param last_support: last support sample
        :param episode_hparams: N, K, Q
        :return:
        """

        last_support = supports[-1]

        # add to the last sample a prototype node for each class
        last_support.num_nodes += episode_hparams.num_classes_per_episode

        feature_dim = last_support.x.shape[-1]
        # initialize the prototype features as ones
        if artificial_node_features == "ones":
            prototype_features = torch.ones((episode_hparams.num_classes_per_episode, feature_dim)).type_as(
                last_support.x
            )
        elif artificial_node_features == "zeros":
            prototype_features = torch.zeros((episode_hparams.num_classes_per_episode, feature_dim)).type_as(
                last_support.x
            )
        elif artificial_node_features == "mean":
            support_features = torch.cat([support.x for support in supports], dim=0)
            prototype_features = torch.mean(support_features, dim=0).unsqueeze(0)
            prototype_features = prototype_features.repeat((episode_hparams.num_classes_per_episode, 1))
        else:
            raise NotImplementedError(f"Node features {artificial_node_features} not implemented.")

        last_support.x = torch.cat((last_support.x, prototype_features), dim=0)

    @classmethod
    def get_label_to_prototype_mapping(cls, episode: Episode, cumulative_node_count: int):
        """
        Assign to each global label in the episode a prototype node index, which will be used
        to index the corresponding embedding matrix to obtain the class prototype embedding

        :param episode: current episode
        :param cumulative_node_count: total number of nodes of the previous episodes
        :return episode_label_to_prot: mapping global label -> corresponding prototype node index
        """

        # assign each global label to a prototype index, these are picked among the indices of the last
        # N nodes of the last sample, which are also the last N nodes of all the supports in the episode
        # first label obtains cumulative_node_count-1, the second cumulative_node_count-2 and so on
        episode_label_to_prot = {
            global_label: cumulative_node_count - (ind + 1) for ind, global_label in enumerate(episode.global_labels)
        }

        return episode_label_to_prot

    @classmethod
    def get_aggregator_prototype_edges(
        cls, episode: Episode, label_to_prototype_mapping: Dict, episode_hparams: EpisodeHParams, episode_cumsum: int
    ) -> torch.Tensor:
        """

        :param episode: current episode
        :param label_to_prototype_mapping: mapping global label -> corresponding prototype node index
        :param episode_hparams: N, K, Q
        :param episode_cumsum: cumulative number of nodes of the graphs in the previous episodes

        :return: artificial aggregator -> prototype edge index
        """

        supports: List[Data] = episode.supports
        N = episode_hparams.num_classes_per_episode

        aggregator_to_prototype_edges = []

        # cumulative node count of support graphs in the episode
        local_cumulative_node_count = 0
        for ind, support in enumerate(supports):
            label_prototype_node = label_to_prototype_mapping[support.y.item()]

            local_cumulative_node_count += support.num_nodes

            # for each support, the aggregator node is the last node in the graph,
            global_cumulative_node_count = episode_cumsum + local_cumulative_node_count
            aggregator_node_index = global_cumulative_node_count - 1

            # take into account the N prototype nodes at the last support
            last_support = ind == (len(supports) - 1)
            if last_support:
                aggregator_node_index -= N

            pooling_to_prototype_edge = [aggregator_node_index, label_prototype_node]

            aggregator_to_prototype_edges.append(pooling_to_prototype_edge)

        return torch.tensor(aggregator_to_prototype_edges).transpose(1, 0)

    @classmethod
    def update_supports_batch_with_prototype_edges(
        cls,
        supports_batch: Batch,
        episode_list: List[Episode],
        all_prototype_edges: List[torch.Tensor],
        episode_hparams: EpisodeHParams,
    ):
        """
        For each aggregator a with class c, add the directed edge (aggregator, prototype_c) to the
        corresponding class prototype node.
        Update the supports batch in place.

        :param supports_batch: Batch containing the support samples
        :param episode_list: List (num_episodes) containing the Episodes
        :param all_prototype_edges: List (num_episodes) containing the prototype edges for each episode
        :param episode_hparams: N, K, Q
        :return:
        """
        supports_by_episodes = [Batch.from_data_list(episode.supports) for episode in episode_list]

        batch_edge_index = []
        cumsum = 0
        for episode_ind, episode_supports in enumerate(supports_by_episodes):
            episode_edges = episode_supports.edge_index
            prototype_edges = all_prototype_edges[episode_ind]

            # update the node indices so to refer to the whole batch
            episode_edges += cumsum
            cumsum += episode_supports.num_nodes

            # add the prototype edges to the other edges
            episode_edges = torch.cat((episode_edges, prototype_edges), dim=1)

            batch_edge_index.append(episode_edges)

        batch_edge_index = torch.cat(batch_edge_index, dim=1)

        supports_batch.edge_index = batch_edge_index
        supports_batch.num_edges += episode_hparams.num_classes_per_episode

    def get_aggregator_indices(self, queries_or_supports: str):
        """
        Retrieves the indices of the aggregator nodes.

        :param queries_or_supports: whether query or support samples
        :return: indices of the aggregator nodes
        """
        # for queries, the aggregator node of a sample is just its last node
        if queries_or_supports == "queries":
            return self.queries.ptr[1:] - 1

        # for supports, the aggregator node of a sample is the last node except that for the last sample
        else:
            ptr = self.supports.ptr[1:] - 1
            aggregator_indices_by_episode = ptr.split(tuple([self.num_supports_per_episode] * self.num_episodes))

            # for the last sample of each episode, the aggregator is not the last node as
            # the last N are prototype nodes instead
            for episode_aggregators in aggregator_indices_by_episode:
                episode_aggregators[-1] -= self.episode_hparams.num_classes_per_episode

            aggregator_indices = torch.cat(aggregator_indices_by_episode, dim=0)

            return aggregator_indices

    def get_global_labels_by_episode(self):
        return self.global_labels.split([self.episode_hparams.num_classes_per_episode] * self.num_episodes)

    def get_support_labels_by_episode(self):
        return self.supports.y.split([self.num_supports_per_episode] * self.num_episodes)

    def get_query_labels_by_episode(self):
        return self.queries.y.split(tuple([self.num_queries_per_episode] * self.num_episodes))

    @property
    def feature_dim(self):
        assert self.supports.x.shape[-1] == self.queries.x.shape[-1]
        return self.supports.x.shape[-1]

    def plot_batch(self, batch: Batch, supports_or_queries: str):
        """
        Plot a batch of either support or query samples, used to validate aggregator
        and prototypal nodes and edges.

        :param batch: batched queries or support graphs
        :param supports_or_queries: whether supports or queries
        :return:
        """

        g = to_networkx(batch, to_undirected=False)

        pos = nx.nx_agraph.graphviz_layout(g, prog="twopi", args="")
        plt.figure(figsize=(16, 16))

        aggregator_nodes = [node.item() for node in self.get_aggregator_indices(supports_or_queries)]
        all_nodes = set(range(batch.num_nodes))
        normal_nodes = all_nodes.difference(aggregator_nodes)

        normal_nodes_labels = {k: k for k, v in pos.items() if k in normal_nodes}
        aggregator_nodes_labels = {k: k for k, v in pos.items() if k in aggregator_nodes}

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

        nx.draw_networkx_nodes(g, pos, nodelist=normal_nodes, node_color="tab:grey", node_size=40, alpha=0.5)
        nx.draw_networkx_labels(g, pos, labels=normal_nodes_labels, alpha=0.5)
        nx.draw_networkx_edges(g, pos, edgelist=normal_edges, edge_color="tab:grey")

        nx.draw_networkx_nodes(g, pos, nodelist=aggregator_nodes, node_color="tab:blue", node_size=80, alpha=0.5)
        nx.draw_networkx_labels(g, pos, labels=aggregator_nodes_labels, alpha=0.5)
        nx.draw_networkx_edges(g, pos, edgelist=aggregator_edges, edge_color="tab:blue", style="dashed")

        if supports_or_queries == "supports":
            prototype_nodes_labels = {k: k for k, v in pos.items() if k in prototype_nodes}

            nx.draw_networkx_nodes(g, pos, nodelist=prototype_nodes, node_color="#FAC60E", node_size=160, alpha=0.5)
            nx.draw_networkx_labels(g, pos, labels=prototype_nodes_labels, alpha=0.5)
            nx.draw_networkx_edges(g, pos, edgelist=prototype_edges, edge_color="#FAC60E", style="dashed")

        plt.axis("equal")

        plt.show()

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
