import json
import logging
from pathlib import Path
from typing import Dict

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self, class_to_label_dict, feature_dim, num_classes_per_episode: int, classes_split: Dict):
        """
        The data information the Lightning Module will be provided with.
        This is a "bridge" between the Lightning DataModule and the Lightning Module.
        There is no constraint on the class name nor in the stored information, as long as it exposes the
        `save` and `load` methods.
        The Lightning Module will receive an instance of MetaData when instantiated,
        both in the train loop or when restored from a checkpoint.
        This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
        DataModule/Trainer independent (useful in prediction scenarios).
        MetaData should contain all the information needed at test time, derived from its train dataset.
        Examples are the class names in a classification task or the vocabulary in NLP tasks.
        MetaData exposes `save` and `load`. Those are two user-defined methods that specify
        how to serialize and de-serialize the information contained in its attributes.
        This is needed for the checkpointing restore to work properly.

        :param class_to_label_dict
        :param feature_dim
        :param num_classes_per_episode
        :param classes_split
        """
        self.classes_to_label_dict = class_to_label_dict
        self.feature_dim = feature_dim
        self.num_classes = len(class_to_label_dict)
        self.num_classes_per_episode = num_classes_per_episode
        self.classes_split = classes_split

    def save(self, dst_path: Path) -> None:
        """
        Serialize the MetaData attributes into the zipped checkpoint in dst_path.

        :param dst_path: the root folder of the metadata inside the zipped checkpoint

        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        data = {
            "classes_to_label_dict": self.classes_to_label_dict,
            "feature_dim": self.feature_dim,
            "num_classes_per_episode": self.num_classes_per_episode,
            "classes_split": self.classes_split,
        }

        (dst_path / "data.json").write_text(json.dumps(data, indent=4, default=lambda x: x.__dict__))

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """
        Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.

        :param src_path: the root folder of the metadata inside the zipped checkpoint

        :return an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        data = json.loads((src_path / "data.json").read_text(encoding="utf-8"))

        return MetaData(
            class_to_label_dict=data["classes_to_label_dict"],
            feature_dim=data["feature_dim"],
            num_classes_per_episode=data["num_classes_per_episode"],
            classes_split=data["classes_split"],
        )
