import os
import pickle
from typing import Dict, List

import networkx as nx
import numpy as np
import scipy
import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Data


class Node:
    def __init__(self, tag, neighbors, attrs):
        self.tag = tag
        self.neighbors = neighbors
        self.attrs = attrs


def load_data(dir_path, dataset_name, feature_params, add_aggregator_nodes, artificial_node_features):
    """
    Loads a TU graph dataset.

    :param dir_path: path to the directory containing the dataset
    :param dataset_name: name of the dataset
    :param feature_params: params regarding data features
    :return:
    """

    graph_list = load_graph_list(dir_path, dataset_name)

    if "pos_enc" in feature_params["features_to_consider"]:
        add_positional_encoding(graph_list)

    class_to_label_dict = get_classes_to_label_dict(graph_list)
    data_list = to_data_list(graph_list, class_to_label_dict, feature_params, add_aggregator_nodes)

    set_node_features(
        data_list,
        feature_params=feature_params,
        add_aggregator_nodes=add_aggregator_nodes,
        artificial_node_features=artificial_node_features,
    )

    return data_list, class_to_label_dict


def add_positional_encoding(graph_list: List[nx.Graph], num_features=5):

    for graph in graph_list:

        positional_encodings = get_positional_encoding_from_nx(graph, num_features=num_features)

        graph.graph["pos_enc"] = positional_encodings


def get_positional_encoding_from_nx(graph, num_features):
    laplacian = nx.laplacian_matrix(graph).asfptype()

    if graph.number_of_nodes() > 2:
        eigenvals, eigenvecs = scipy.sparse.linalg.eigsh(laplacian, k=2, which="SM")
        principal_eigenvec = eigenvecs.transpose()[1]
    else:
        laplacian = laplacian.todense()
        eigenvals, eigenvecs = np.linalg.eig(laplacian)
        principal_eigenvec = np.array(eigenvecs.transpose()[1]).squeeze(axis=0)

    positional_encodings = []
    for k in range(1, num_features + 1):
        kth_pos_enc = np.cos(principal_eigenvec * 2 * np.pi * k)
        positional_encodings.append(kth_pos_enc)

    positional_encodings = np.stack(positional_encodings, axis=1)

    return torch.tensor(positional_encodings, dtype=torch.float32)


def load_graph_list(dir_path, dataset_name):
    """
    Loads a graph dataset as a list of networkx graphs

    :param dir_path: path to the directory containing the dataset
    :param dataset_name: name of the dataset

    :return: graph_list: list of networkx graphs
    """
    dataset_path = f"{os.path.join(dir_path, dataset_name)}.txt"

    graph_list = []
    with open(dataset_path, "r") as f:
        num_graphs = int(f.readline().strip())

        for graph_ind in range(num_graphs):

            graph: nx.Graph = parse_graph(f)
            graph_list.append(graph)

    return graph_list


def to_data_list(graph_list, class_to_label_dict, feature_params, add_aggregator_nodes) -> List[Data]:
    """
    Converts a list of Networkx graphs to a list of PyG Data objects

    :param graph_list: list of Networkx graphs
    :param class_to_label_dict: mapping original class to integer label

    :return:
    """
    data_list = []

    for G in graph_list:
        edge_index = get_edge_index_from_nx(G, add_aggregator_nodes)
        label = torch.tensor(class_to_label_dict[G.graph["class"]], dtype=torch.long).unsqueeze(0)

        data_args = {
            "edge_index": edge_index,
            "num_nodes": G.number_of_nodes(),
            "y": label,
            "degrees": get_degree_tensor_from_nx(G),
            "tags": get_tag_tensor_from_nx(G),
        }

        if "pos_enc" in feature_params["features_to_consider"]:
            data_args["pos_enc"] = G.graph["pos_enc"].transpose(1, 0)

        if "num_cycles" in feature_params["features_to_consider"]:
            data_args["num_cycles"] = (get_num_cycles_from_nx(G, feature_params["max_considered_cycle_len"]),)

        data = Data(**data_args)

        data_list.append(data)

    return data_list


def parse_graph(file_descriptor):
    """
    Parses a single graph from file

    :param file_descriptor: file formatted accordingly to TU datasets

    :return: networkx graph
    """

    graph_header = file_descriptor.readline().strip().split()
    num_nodes, cls = [int(w) for w in graph_header]

    G = nx.Graph()
    G.graph["class"] = str(cls)

    for node_ind in range(num_nodes):

        node: Node = parse_node(file_descriptor)

        G.add_node(node_ind, tag=node.tag, attrs=node.attrs)

        for neighbor in node.neighbors:
            G.add_edge(node_ind, neighbor)
            G.add_edge(neighbor, node_ind)

    assert len(G) == num_nodes

    return G


def parse_node(file_descriptor):
    """
    Parses a single node from file, corresponding to a row having format
        tag num_neighbors nghbr_1 nghbr_2 ... attr_1 attr_2 ...

    :param file_descriptor: file formatted accordingly to TU datasets
    :return: Node with tag, neighbors list and possibly attributes
    """

    node_row = file_descriptor.readline().strip().split()

    node_header = node_row[0:2]
    tag, num_neighbors = int(node_header[0]), int(node_header[1])

    # attributes come after the header (tag and num_neighbors) and all the neighbors
    attr_starting_index = 2 + num_neighbors

    neighbors = [int(w) for w in node_row[2:attr_starting_index]]

    attrs = [float(w) for w in node_row[attr_starting_index:]]
    attrs = np.array(attrs) if attrs else None

    return Node(tag, neighbors, attrs)


def get_degree_tensor_from_nx(G: nx.Graph) -> Tensor:
    """
    Returns node degrees as a tensor
    :param G: networkx graph

    :return: tensor ~ (num_nodes) with tensor[i] = degree of node i
    """
    degree_list = sorted(list(G.degree), key=lambda x: x[0])

    return torch.tensor([pair[1] for pair in degree_list])


def get_tag_tensor_from_nx(G: nx.Graph) -> Tensor:
    """
    Returns node tags as a tensor
    :param G: networkx graph

    :return: tensor ~ (num_nodes) with tensor[i] = tag of node i
    """

    tag_dict = nx.get_node_attributes(G, "tag")
    tag_tuples = [(key, value) for key, value in tag_dict.items()]

    node_and_tags_sorted_by_node = sorted(tag_tuples, key=lambda t: t[0])
    tags_sorted_by_node = [tup[1] for tup in node_and_tags_sorted_by_node]

    return torch.tensor(tags_sorted_by_node)


def get_num_cycles_from_nx(G: nx.Graph, max_considered_cycle_len) -> Tensor:

    A = torch.Tensor(nx.adjacency_matrix(G).todense())
    A_k = torch.clone(A)

    num_cycles = []
    for k in range(2, max_considered_cycle_len + 1):
        A_k = A_k.t() @ A
        num_cycles_len_k = A_k.diagonal()

        num_cycles.append(num_cycles_len_k)

    return torch.stack(num_cycles, dim=0)


def set_node_features(
    data_list: List[Data], feature_params: Dict, add_aggregator_nodes: bool = False, artificial_node_features: str = ""
):
    """
    Adds to each data in data_list either the tags, the degrees or both as node features
    In place function

    :param data_list: list of preprocessed graphs as PyG Data objects
    :param feature_params:

    """

    # contains for each graph G its node features, where each feature is a vector of length N_G
    all_node_features = []

    if "tag" in feature_params["features_to_consider"]:
        all_tags = torch.cat([data.tags for data in data_list], 0)
        one_hot_tags = get_one_hot_attrs(all_tags, data_list)
        all_node_features = initialize_or_concatenate(all_node_features, one_hot_tags)

    if "degree" in feature_params["features_to_consider"]:
        all_degrees = torch.cat([data.degrees for data in data_list], 0)
        one_hot_degrees = get_one_hot_attrs(all_degrees, data_list)
        all_node_features = initialize_or_concatenate(all_node_features, one_hot_degrees)

    if "pos_enc" in feature_params["features_to_consider"]:
        for k in range(feature_params["num_pos_encs"]):
            pos_encs = [data.pos_enc[k].unsqueeze(1) for data in data_list]
            all_node_features = initialize_or_concatenate(all_node_features, pos_encs)

    if "num_cycles" in feature_params["features_to_consider"]:
        for k in range(1, feature_params["max_considered_cycle_len"]):
            num_cycles = [data.num_cycles[k].unsqueeze(1) for data in data_list]
            all_node_features = initialize_or_concatenate(all_node_features, num_cycles)

    for data, node_features in zip(data_list, all_node_features):
        assert data.num_nodes == node_features.shape[0]
        if add_aggregator_nodes:

            if artificial_node_features == "zeros":
                aggregator_node_features = torch.zeros_like(node_features[0]).unsqueeze(0)
            elif artificial_node_features == "ones":
                aggregator_node_features = torch.ones_like(node_features[0]).unsqueeze(0)
            elif artificial_node_features == "mean":
                aggregator_node_features = torch.mean(node_features, dim=0).unsqueeze(0)
            else:
                raise NotImplementedError(f"Node features {artificial_node_features} not implemented")

            data.num_nodes = data.num_nodes + 1
            node_features = torch.cat((node_features, aggregator_node_features), dim=0)

        data["x"] = node_features
        # TODO: check if this must be updated when adding aggregator edges
        data["num_sample_edges"] = data.edge_index.shape[1]
        data["degrees"] = None
        data["tags"] = None
        data["num_cycles"] = None
        data["pos_enc"] = None


def initialize_or_concatenate(all_node_features, feature_to_add):

    if len(all_node_features) == 0:
        return feature_to_add

    num_graphs = len(all_node_features)

    new_all_node_features = [torch.cat((all_node_features[i], feature_to_add[i]), dim=1) for i in range(num_graphs)]

    return new_all_node_features


def get_one_hot_attrs(attrs, data_list):
    """

    :param attrs:
    :param data_list:
    :return:
    """
    # unique_attrs contains the unique values found in attrs,
    # corrs contains the indices of the unique array that reconstruct the input array
    unique_attrs, corrs = np.unique(attrs, return_inverse=True, axis=0)
    num_different_attrs = len(unique_attrs)

    # encode
    all_one_hot_attrs = []
    pointer = 0

    for data in data_list:
        hots = torch.LongTensor(corrs[pointer : pointer + data.num_nodes])
        data_one_hot_attrs = F.one_hot(hots, num_different_attrs).float()

        all_one_hot_attrs.append(data_one_hot_attrs)
        pointer += data.num_nodes

    return all_one_hot_attrs


def get_edge_index_from_nx(G: nx.Graph, add_aggregator_nodes=False) -> Tensor:
    """
    Extracts edge index from networkx graph
    :param G: networkx graph
    :return: tensor ~ (2, num_edges) containing all the edges in the graph G
    """
    # shape (num_edges*2, 2)
    edges_tensor = torch.tensor(list([(edge[0], edge[1]) for edge in G.edges]), dtype=torch.long)
    edges_tensor_reverse = torch.tensor(list([(edge[1], edge[0]) for edge in G.edges]), dtype=torch.long)

    edge_index = torch.cat((edges_tensor, edges_tensor_reverse), dim=0)

    # aggregator node is in the last index
    aggregator_node_index = G.number_of_nodes()
    if add_aggregator_nodes:
        aggregator_edges = torch.tensor([(node, aggregator_node_index) for node in range(0, aggregator_node_index)])
        edge_index = torch.cat((edge_index, aggregator_edges), dim=0)

    return edge_index.t().contiguous()


def get_classes_to_label_dict(graph_list) -> Dict:
    """
    Obtains all the classes present in the data and maps them to progressive integers.

    :param graph_list: list of networkx graphs

    :return: map that maps each string class to an integer
    """

    all_classes = {graph.graph["class"] for graph in graph_list}
    all_classes_sorted = sorted([int(cls) for cls in all_classes])
    class_to_label_dict = {str(cls): label for label, cls in enumerate(all_classes_sorted)}

    return class_to_label_dict


def load_pickle_data(data_dir, dataset_name, feature_params, add_aggregator_nodes, artificial_node_features):

    node_attrs = load_pickle(os.path.join(data_dir, dataset_name + "_node_attributes.pickle"))
    base_set = load_pickle(os.path.join(data_dir, dataset_name + "_base.pickle"))
    novel_set = load_pickle(os.path.join(data_dir, dataset_name + "_novel.pickle"))
    val_set = load_pickle(os.path.join(data_dir, dataset_name + "_val_set.pickle"))

    classes_split = {
        "base": sorted(list(base_set["label2graphs"].keys())),
        "val": sorted(list(val_set["label2graphs"].keys())),
        "novel": sorted(list(novel_set["label2graphs"].keys())),
    }

    data_list = graph_dict_to_data_list(
        base_set, node_attrs, feature_params, add_aggregator_nodes, artificial_node_features
    )
    data_list += graph_dict_to_data_list(
        val_set, node_attrs, feature_params, add_aggregator_nodes, artificial_node_features
    )
    data_list += graph_dict_to_data_list(
        novel_set, node_attrs, feature_params, add_aggregator_nodes, artificial_node_features
    )
    assert len(data_list) == len(base_set["graph2nodes"]) + len(val_set["graph2nodes"]) + len(novel_set["graph2nodes"])

    return data_list, classes_split


def graph_dict_to_data_list(graph_set, node_attrs, feature_params, add_aggregator_nodes, artificial_node_features):
    data_list = []

    for cls, graph_indices in graph_set["label2graphs"].items():

        for graph_idx in graph_indices:

            nodes_global_to_local_map = {
                global_idx: local_idx for local_idx, global_idx in enumerate(graph_set["graph2nodes"][graph_idx])
            }
            edge_indices = torch.tensor(graph_set["graph2edges"][graph_idx], dtype=torch.long)
            edge_indices.apply_(lambda val: nodes_global_to_local_map.get(val))

            num_nodes = len(nodes_global_to_local_map)

            if add_aggregator_nodes:
                aggregator_node_index = num_nodes
                num_nodes += 1
                aggregator_edges = torch.tensor(
                    [(node, aggregator_node_index) for node in range(0, aggregator_node_index)]
                )
                edge_indices = torch.cat((edge_indices, aggregator_edges), dim=0)

            edge_index = edge_indices.t().contiguous()

            G = create_networkx_graph(num_nodes, edge_indices)

            node_features = get_node_features(
                graph_set=graph_set,
                node_attrs=node_attrs,
                graph_idx=graph_idx,
                add_aggregator_nodes=add_aggregator_nodes,
                artificial_node_features=artificial_node_features,
            )
            assert node_features.size(0) == num_nodes

            if "num_cycles" in feature_params["features_to_consider"]:
                num_cycles = get_num_cycles_from_nx(
                    G, max_considered_cycle_len=feature_params["max_considered_cycle_len"]
                )
                node_features = torch.cat((node_features, num_cycles.t()), dim=1)

            if "pos_enc" in feature_params["features_to_consider"]:
                pos_enc = get_positional_encoding_from_nx(G, num_features=feature_params["num_pos_encs"])
                node_features = torch.cat((node_features, pos_enc), dim=1)

            data = Data(
                x=node_features,
                edge_index=edge_index,
                num_nodes=num_nodes,
                y=torch.tensor(cls, dtype=torch.long),
            )

            data_list.append(data)

    return data_list


# def graph_dict_to_data_list(graph_set, node_attrs, feature_params, add_aggregator_nodes, artificial_node_features):
#     data_list = []
#
#     max_num_cycles_length_k = {k: 0 for k in range(2, feature_params["max_considered_cycle_len"] + 1)}
#
#     for cls, graph_indices in graph_set["label2graphs"].items():
#
#         for graph_idx in graph_indices:
#
#             nodes_global_to_local_map = {
#                 global_idx: local_idx for local_idx, global_idx in enumerate(graph_set["graph2nodes"][graph_idx])
#             }
#             edge_indices = torch.tensor(graph_set["graph2edges"][graph_idx], dtype=torch.long)
#             edge_indices.apply_(lambda val: nodes_global_to_local_map.get(val))
#
#             num_nodes = len(nodes_global_to_local_map)
#
#             edge_index = edge_indices.t().contiguous()
#
#             G = create_networkx_graph(num_nodes, edge_indices)
#
#             node_features = get_node_features(
#                 graph_set=graph_set,
#                 node_attrs=node_attrs,
#                 graph_idx=graph_idx,
#                 add_aggregator_nodes=add_aggregator_nodes,
#                 artificial_node_features=artificial_node_features,
#             )
#             assert node_features.size(0) == num_nodes
#
#             # TODO: remove
#             feature_dim = node_features.size(1)
#
#             if "num_cycles" in feature_params["features_to_consider"]:
#                 num_cycles = get_num_cycles_from_nx(
#                     G,
#                     max_considered_cycle_len=feature_params["max_considered_cycle_len"],
#                     max_num_cycles_length_k=max_num_cycles_length_k,
#                 )
#                 node_features = torch.cat((node_features, num_cycles.t()), dim=1)
#
#             if "pos_enc" in feature_params["features_to_consider"]:
#                 pos_enc = get_positional_encoding_from_nx(G, num_features=feature_params["num_pos_encs"])
#                 node_features = torch.cat((node_features, pos_enc), dim=1)
#
#             data = Data(
#                 x=node_features,
#                 edge_index=edge_index,
#                 num_nodes=num_nodes,
#                 y=torch.tensor(cls, dtype=torch.long),
#             )
#
#             data_list.append(data)
#
#     K = feature_params["max_considered_cycle_len"]
#     for data in data_list:
#
#         for ind, k in enumerate(range(2, K + 1)):
#             data.x[:, ind + feature_dim] = data.x[:, ind + feature_dim] / max_num_cycles_length_k[k]
#
#     return data_list


def create_networkx_graph(num_nodes, edge_indices):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for edge in edge_indices:
        u, v = edge[0].item(), edge[1].item()
        G.add_edge(u, v)
        G.add_edge(v, u)

    return G


def get_node_features(graph_set, node_attrs, graph_idx, add_aggregator_nodes, artificial_node_features):
    node_features = []

    for node in graph_set["graph2nodes"][graph_idx]:
        attr = torch.tensor(node_attrs[node], dtype=torch.float)
        if len(attr.size()) == 0:
            attr = attr.unsqueeze(0)
        node_features.append(attr)

    if add_aggregator_nodes:
        if artificial_node_features == "ones":
            aggregator_features = torch.ones_like(node_features[0])
        elif artificial_node_features == "zeros":
            aggregator_features = torch.zeros_like(node_features[0])
        elif artificial_node_features == "mean":
            aggregator_features = torch.mean(torch.cat(node_features, dim=0), dim=0).unsqueeze(0)
        else:
            raise NotImplementedError(f"Node features {artificial_node_features} not implemented")
        node_features.append(aggregator_features)

    return torch.stack(node_features)


def load_pickle(file_name):
    with open(file_name, "rb") as f:
        data = pickle.load(f)
        return data


def load_query_support_idxs(path):
    raise NotImplementedError
