from typing import List

from grakel.utils import graph_from_networkx
from torch_geometric.data import Batch, Data

from fs_grl.data.io_utils import data_list_to_graph_list


def batch_to_grakel(batch: Batch, node_labels_tag):
    data_list: List[Data] = batch.to_data_list()
    graph_list = data_list_to_graph_list(data_list)
    return graph_from_networkx(graph_list, node_labels_tag=node_labels_tag)
