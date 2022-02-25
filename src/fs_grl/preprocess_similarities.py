import logging
from pathlib import Path
from typing import Dict, List

import hydra
import omegaconf
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.serialization import load_model

import fs_grl  # noqa

# Force the execution of __init__.py if this file is executed directly.
from fs_grl.data.datamodule import GraphFewShotDataModule
from fs_grl.data.dataset import VanillaGraphDataset
from fs_grl.pl_modules.distance_metric_learning import DistanceMetricLearning

pylogger = logging.getLogger(__name__)


def compute_global_prototypes(
    model, data_list_by_label: Dict[int, List[Data]], label_to_class_dict: Dict[int, str]
) -> Dict[str, torch.Tensor]:
    """
    Computes the prototype for each label in the dataset by averaging the samples of that label.
    :param model: pretrained model
    :param data_list_by_label: samples grouped by label
    :param label_to_class_dict: mapping label -> class
    :return:
    """
    prototypes = {}

    for label, data_list in data_list_by_label.items():

        dataset = VanillaGraphDataset(data_list)
        dataloader = DataLoader(dataset=dataset, collate_fn=Batch.from_data_list, batch_size=64)

        all_label_embeddings = []
        for batch in dataloader:
            batch_embeddings = model.model._embed(batch)
            all_label_embeddings.append(batch_embeddings)

        all_label_embeddings = torch.cat(all_label_embeddings, dim=0)
        label_prototype = torch.mean(all_label_embeddings, dim=0)

        cls = label_to_class_dict[label]
        prototypes[cls] = label_prototype

    return prototypes


def run(cfg: DictConfig):
    """Generic train loop.
    Args:
        cfg: run configuration, defined by Hydra in /conf
    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    seed_index_everything(cfg.train)

    pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")
    datamodule: GraphFewShotDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    pylogger.info("Loading pretrained model")

    model = load_model(
        module_class=DistanceMetricLearning,
        checkpoint_path=Path(cfg.nn.data.best_model_path),
    )
    model.eval()

    data_list_by_base_label = {
        label: data_list
        for label, data_list in datamodule.data_list_by_label.items()
        if label in datamodule.base_labels
    }

    prototypes = compute_global_prototypes(model, data_list_by_base_label, datamodule.label_to_class_dict)

    torch.save(prototypes, cfg.nn.data.prototypes_path)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
