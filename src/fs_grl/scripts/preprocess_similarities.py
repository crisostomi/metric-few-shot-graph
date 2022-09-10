import logging
from pathlib import Path

import hydra
import omegaconf
import torch
from omegaconf import DictConfig

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.serialization import load_model

import fs_grl  # noqa

# Force the execution of __init__.py if this file is executed directly.
from fs_grl.data.datamodule.datamodule import GraphFewShotDataModule
from fs_grl.pl_modules.distance_metric_learning import DistanceMetricLearning
from fs_grl.utils import compute_global_prototypes

pylogger = logging.getLogger(__name__)


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

    prototypes = compute_global_prototypes(model, datamodule.data_list_by_base_label, datamodule.id_to_property)

    torch.save(prototypes, cfg.nn.data.prototypes_path)
    torch.save(model.model.metric_scaling_factor, cfg.nn.data.scaling_factor_path)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
