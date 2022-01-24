import pytest
import torch

from nn_core.common import PROJECT_ROOT

from fs_grl.data.io_utils import load_graph_list, to_data_list


@pytest.fixture()
def graph_list():
    dir_path, dataset_name = str(PROJECT_ROOT / "data" / "DUMMY"), "DUMMY"
    graph_list = load_graph_list(dir_path, dataset_name)
    return graph_list


def test_load_graph_list(graph_list):
    """
    DUMMY contains
        1. house graph
        2. kite graph
        3. union of a triangle, an edge and a vertex
    :return:
    """
    graph_names = "house", "kite", "triangle_edge_vertex"

    graphs = {}
    graphs["house"], graphs["kite"], graphs["triangle_edge_vertex"] = graph_list
    number_of_nodes = {"house": 5, "kite": 5, "triangle_edge_vertex": 6}
    number_of_edges = {"house": 6, "kite": 6, "triangle_edge_vertex": 4}

    for graph_name in graph_names:
        assert graphs[graph_name].number_of_nodes() == number_of_nodes[graph_name]
        assert graphs[graph_name].number_of_edges() == number_of_edges[graph_name]


@pytest.fixture
def data_list(graph_list):
    return to_data_list(graph_list)


def test_to_data_list(data_list):
    graph_names = "house", "kite", "triangle_edge_vertex"

    data = {}
    data["house"], data["kite"], data["triangle_edge_vertex"] = data_list

    edge_indices = {}
    edge_indices["house"] = torch.tensor([[0, 0, 1, 2, 2, 3], [1, 2, 3, 3, 4, 4]])
    edge_indices["kite"] = torch.tensor([[0, 0, 1, 1, 2, 3], [1, 2, 2, 3, 3, 4]])
    edge_indices["triangle_edge_vertex"] = torch.tensor([[0, 0, 1, 3], [1, 2, 2, 4]])

    for graph_name in graph_names:
        assert torch.equal(edge_indices[graph_name], data[graph_name].edge_index)


# TODO: fix
# def test_add_node_degrees_as_tags(data_list):
#
#     add_node_degrees_as_tags(data_list)
#
#     graph_names = "house", "kite", "triangle_edge_vertex"
#
#     data = {}
#     data["house"], data["kite"], data["triangle_edge_vertex"] = data_list
#
#     degree_lists = {}
#     degree_lists["house"] = torch.tensor([2, 2, 3, 3, 2])
#     degree_lists["kite"] = torch.tensor([2, 3, 3, 3, 1])
#     degree_lists["triangle_edge_vertex"] = torch.tensor([2, 2, 2, 1, 1, 0])
#
#     degree_one_hot = {}
#
#     degree_one_hot["house"] = torch.tensor(
#         [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
#     )
#
#     degree_one_hot["kite"] = torch.tensor(
#         [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]]
#     )
#
#     degree_one_hot["triangle_edge_vertex"] = torch.tensor(
#         [
#             [0.0, 0.0, 1.0, 0.0],
#             [0.0, 0.0, 1.0, 0.0],
#             [0.0, 0.0, 1.0, 0.0],
#             [0.0, 1.0, 0.0, 0.0],
#             [0.0, 1.0, 0.0, 0.0],
#             [1.0, 0.0, 0.0, 0.0],
#         ]
#     )
#
#     for graph_name in graph_names:
#         assert torch.equal(data[graph_name].degrees, degree_lists[graph_name])
#         assert torch.equal(data[graph_name].x, degree_one_hot[graph_name])
