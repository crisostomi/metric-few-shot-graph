import json
import random
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
from grakel.utils import graph_from_networkx

from fs_grl.data.io_utils import get_classes_to_label_dict, load_graph_list, map_classes_to_labels
from fs_grl.data.utils import get_label_to_samples_map


def fetch_dataset(cfg):
    data_dir = cfg.data_dir
    dataset_name = cfg.dataset_name
    classes_split_path = cfg.classes_split_path

    if dataset_name in {"TRIANGLES", "ENZYMES", "Letter_high", "Reddit", "DUMMY"}:
        classes_split = json.loads(Path(classes_split_path).read_text(encoding="utf-8"))
        graph_list: List[nx.Graph] = load_graph_list(data_dir, dataset_name)
        class_to_label_dict = get_classes_to_label_dict(graph_list)
        map_classes_to_labels(graph_list, class_to_label_dict)

        labels_per_split = labels_split(class_to_label_dict, classes_split)

        graph_list_by_label: Dict[int, List[nx.Graph]] = get_label_to_samples_map(graph_list)
        G_train = [
            graph
            for label, graph_list in graph_list_by_label.items()
            for graph in graph_list
            if label in labels_per_split["base"]
        ]
        random.shuffle(G_train)
        y_train = np.array([graph.graph["class"] for graph in G_train])

        G_test = {
            label: graph_list for label, graph_list in graph_list_by_label.items() if label in labels_per_split["novel"]
        }
        episodes = [sample_episode(cfg, G_test, labels_per_split) for _ in range(cfg.num_episodes_per_epoch.test)]

        return graph_from_networkx(G_train, node_labels_tag="tag"), y_train, episodes

    elif dataset_name in {"COIL-DEL", "R52"}:
        pass
    else:
        raise NotImplementedError


def labels_split(class_to_label_dict, classes_split):
    return {
        split: sorted([class_to_label_dict[str(cls)] for cls in classes]) for split, classes in classes_split.items()
    }


def sample_episode(cfg, G_test, labels_per_split):

    episode_labels = sample_labels(cfg, labels_per_split)

    supports = []
    queries = []

    for label in episode_labels:
        label_supports_queries = sample_label_queries_supports(cfg, G_test, label)

        supports += label_supports_queries["supports"]
        queries += label_supports_queries["queries"]

    random.shuffle(supports)
    random.shuffle(queries)

    y_supports = np.array([support.graph["class"] for support in supports])
    y_queries = np.array([query.graph["class"] for query in queries])

    assert len(y_queries) == len(queries) and len(y_supports) == len(supports)

    return {
        "supports": graph_from_networkx(supports, node_labels_tag="tag"),
        "y_supports": y_supports,
        "queries": graph_from_networkx(queries, node_labels_tag="tag"),
        "y_queries": y_queries,
    }


def sample_labels(cfg, labels_per_split) -> List:
    """
    Sample N labels for an episode from the stage labels

    :return
    """
    sampled_labels = random.sample(labels_per_split["novel"], cfg.episode_hparams.test.num_classes_per_episode)
    return sorted(sampled_labels)


def sample_label_queries_supports(cfg, samples_by_label, label: int) -> Dict[str, List]:
    f"""
    Given a label {label}, samples K support and Q queries. These are always disjoint inside the episode.
    """

    all_label_supports: List = samples_by_label[label]

    label_samples_episode = random.sample(
        all_label_supports,
        cfg.episode_hparams.test.num_supports_per_class + cfg.episode_hparams.test.num_queries_per_class,
    )

    label_supports_episode = label_samples_episode[: cfg.episode_hparams.test.num_supports_per_class]
    label_queries_episode = label_samples_episode[cfg.episode_hparams.test.num_supports_per_class :]

    return {"supports": label_supports_episode, "queries": label_queries_episode}
