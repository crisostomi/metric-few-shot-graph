import math
from random import shuffle
from typing import Dict, Iterable, List, Tuple, Union

import networkx as nx
import numpy as np
import torch
from backports.strenum import StrEnum
from torch_geometric.data import Data


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)


class SampleType(StrEnum):
    SUPPORT = "supports"
    QUERY = "queries"


class SupportsAggregation(StrEnum):
    MEAN = "mean"
    DEEPSETS = "deepsets"


def flatten(iterable: Iterable) -> List:
    """ """
    return [el for sublist in iterable for el in sublist]


def random_split_sequence(sequence: List, split_ratio: float) -> Tuple[List, List]:
    f"""
    Splits a sequence randomly into two sequences, the first having {split_ratio}% of the elements
    and the second having {1-split_ratio}%.
    :param sequence: sequence to be split.
    :param split_ratio: percentage of the elements falling in the first sequence.
    :return subseq_1, subseq_2
    """

    idxs = np.arange(len(sequence))
    np.random.shuffle(idxs)

    support_upperbound = math.ceil(split_ratio * len(sequence))
    split_sequence_1_idxs = idxs[:support_upperbound]
    split_sequence_2_idxs = idxs[support_upperbound:]

    split_seq_1 = [sequence[idx] for idx in split_sequence_1_idxs]
    split_seq_2 = [sequence[idx] for idx in split_sequence_2_idxs]

    return split_seq_1, split_seq_2


def random_split_bucketed(sequence: List, split_ratio: float) -> Tuple[List, List]:
    """
    Splits a sequence so to have the same distribution in each bucket
    :param sequence:
    :param split_ratio:
    :return
    """

    sequence_bucketed = get_label_to_samples_map(sequence)

    split_sequence_1 = []
    split_sequence_2 = []

    for key, subseq in sequence_bucketed.items():
        split_subseq_1, split_subseq_2 = random_split_sequence(list(subseq), split_ratio)
        split_sequence_1.append(split_subseq_1)
        split_sequence_2.append(split_subseq_2)

    split_sequence_1 = flatten(split_sequence_1)
    split_sequence_2 = flatten(split_sequence_2)

    shuffle(split_sequence_1)
    shuffle(split_sequence_2)

    return split_sequence_1, split_sequence_2


def get_label_to_samples_map(annotated_samples: List) -> Dict[int, List[Union[Data, nx.Graph]]]:
    """
    Given a list of annotated_samples, return a map { label: list of samples with that label}
    """
    res = {}
    for sample in annotated_samples:
        y = get_label_from_graph(sample)
        res.setdefault(y, []).append(sample)
    return res


def get_label_from_graph(graph: Union[nx.Graph, Data]) -> int:
    if isinstance(graph, nx.Graph):
        return graph.graph["class"]
    else:
        return graph.y.item()


def get_lens_from_batch_assignment(batch_assignment):
    lens = torch.unique(batch_assignment, return_counts=True)[1]

    return lens
