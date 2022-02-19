import os
from typing import List

import networkx as nx
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Data


class Node:
    def __init__(self, tag, neighbors, attrs):
        self.tag = tag
        self.neighbors = neighbors
        self.attrs = attrs


def load_data(dir_path, dataset_name, attr_to_consider):
    """
    Loads a TU graph dataset.

    :param dir_path: path to the directory containing the dataset
    :param dataset_name: name of the dataset
    :param attr_to_consider:
    :return:
    """
    graph_list = load_graph_list(dir_path, dataset_name)

    class_to_label_dict = get_label_dict(graph_list)
    data_list = to_data_list(graph_list, class_to_label_dict)

    set_node_features(data_list, attr_to_consider=attr_to_consider)

    return data_list, class_to_label_dict


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


def to_data_list(graph_list, class_to_label_dict):
    data_list = []
    for G in graph_list:
        edge_index = get_edge_index_from_nx(G)
        label = torch.tensor(class_to_label_dict[G.graph["label"]]).unsqueeze(0).long()
        data = Data(
            edge_index=edge_index,
            num_nodes=G.number_of_nodes(),
            y=label,
            degrees=get_degree_tensor_from_nx(G),
            tags=get_tag_tensor_from_nx(G),
        )
        data_list.append(data)

    return data_list


def parse_graph(file_descriptor):
    """
    Parses a single graph from file
    :param file_descriptor: file formatted accordingly to TU datasets
    :return: networkx graph
    """

    graph_header = file_descriptor.readline().strip().split()
    num_nodes, label = [int(w) for w in graph_header]

    G = nx.Graph()
    G.graph["label"] = str(label)

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


def get_degree_tensor_from_nx(G: nx.Graph):
    """
    Returns node degrees as a list
    :param G: networkx graph
    :return: list of degrees
    """
    degree_list = sorted(list(G.degree), key=lambda x: x[0])
    return torch.tensor([pair[1] for pair in degree_list])


def get_tag_tensor_from_nx(G: nx.Graph):
    tag_dict = nx.get_node_attributes(G, "tag")
    tag_tuples = [(key, value) for key, value in tag_dict.items()]
    tag_tuples_sorted = sorted(tag_tuples, key=lambda t: t[0])
    tags_sorted_by_node = [tup[1] for tup in tag_tuples_sorted]
    return torch.tensor(tags_sorted_by_node)


def set_node_features(data_list: List[Data], attr_to_consider):
    """
    Adds to each data in data_list either the tags, the degrees or both as node features
    :param data_list:
    :param attr_to_consider:
    :return:
    """
    assert attr_to_consider in {"tag", "degree", "both"}
    one_hot_tags, one_hot_degrees = None, None
    if attr_to_consider in {"tag", "both"}:
        all_tags = torch.cat([data.tags for data in data_list], 0)
        one_hot_tags = get_one_hot_attrs(all_tags, data_list)

    if attr_to_consider in {"degree", "both"}:
        all_degrees = torch.cat([data.degrees for data in data_list], 0)
        one_hot_degrees = get_one_hot_attrs(all_degrees, data_list)

    if attr_to_consider == "both":
        all_node_features = [torch.cat((one_hot_tags[i], one_hot_degrees[i]), dim=1) for i in range(len(data_list))]
    else:
        all_node_features = one_hot_tags if one_hot_tags else one_hot_degrees

    for data, node_features in zip(data_list, all_node_features):
        data["x"] = node_features


def get_one_hot_attrs(attrs, data_list):
    # compute unique values
    unique_attrs, corrs = np.unique(attrs, return_inverse=True, axis=0)
    num_different_attrs = len(unique_attrs)

    # encode
    pointer = 0
    all_one_hot_attrs = []

    for data in data_list:
        hots = torch.LongTensor(corrs[pointer : pointer + data.num_nodes])
        data_one_hot_attrs = F.one_hot(hots, num_different_attrs).float()

        all_one_hot_attrs.append(data_one_hot_attrs)
        pointer += data.num_nodes
    return all_one_hot_attrs


def get_edge_index_from_nx(G: nx.Graph):
    """
    Extracts edge index from networkx graph
    :param G:
    :return:
    """
    return torch.tensor(list(G.edges), dtype=torch.long).t().contiguous().view(2, -1)


def get_label_dict(graph_list):
    label_dict = {}
    for graph in graph_list:
        label = graph.graph["label"]
        if label not in label_dict:
            label_dict[label] = len(label_dict)

    return label_dict


# TODO: implement
def load_query_support_idxs(path):
    pass
