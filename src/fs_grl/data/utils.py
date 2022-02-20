import math
import operator
from itertools import groupby
from random import shuffle
from typing import Iterable, List, Tuple

import numpy as np

AnnotatedSample = Tuple[str, int]


def get_local_to_global_class_mapping(global_classes: List):
    """
    :param global_classes:
    :return:
    """
    global_to_local = {}
    for key in sorted(global_classes):
        global_to_local[key] = len(global_to_local)
    local_to_global = {v: k for k, v in global_to_local.items()}
    return local_to_global


def flatten(iterable: Iterable) -> List:
    """ """
    return [el for sublist in iterable for el in sublist]


def random_split_sequence(sequence: List, split_ratio: float) -> Tuple[List, List]:
    f"""
    Splits a sequence randomly into two sequences, the first having {split_ratio}% of the elements
    and the second having {1-split_ratio}%.
    :param sequence: sequence to be split.
    :param split_ratio: percentage of the elements falling in the first sequence.
    :return: subseq_1, subseq_2
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
    :return:
    """

    sequence_bucketed = groupby(sequence, key=operator.attrgetter("y"))

    split_sequence_1 = []
    split_sequence_2 = []

    for key, subseq in sequence_bucketed:
        split_subseq_1, split_subseq_2 = random_split_sequence(list(subseq), split_ratio)
        split_sequence_1.append(split_subseq_1)
        split_sequence_2.append(split_subseq_2)

    split_sequence_1 = flatten(split_sequence_1)
    split_sequence_2 = flatten(split_sequence_2)

    shuffle(split_sequence_1)
    shuffle(split_sequence_2)

    return split_sequence_1, split_sequence_2
