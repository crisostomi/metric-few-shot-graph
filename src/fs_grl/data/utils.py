from typing import List, Tuple

AnnotatedSample = Tuple[str, int]


def get_local_to_global_class_mapping(global_classes: List):
    """
    :param global_keys:
    :return:
    """
    global_to_local = {}
    for key in sorted(global_classes):
        global_to_local[key] = len(global_to_local)
    local_to_global = {v: k for k, v in global_to_local.items()}
    return local_to_global


def flatten(l: List) -> List:
    """ """
    return [el for sublist in l for el in sublist]
