import hashlib
import logging
from typing import Dict

import hydra
import numpy as np
import omegaconf
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from tqdm import tqdm

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger

import fs_grl  # noqa
from fs_grl.custom_pipelines.graph_kernel.utils import fetch_dataset
from fs_grl.utils import handle_fast_dev_run

# Force the execution of __init__.py if this file is executed directly.

pylogger = logging.getLogger(__name__)


def add_run_digest(cfg, keys_to_ignore):
    OmegaConf.set_struct(cfg, True)

    hash_builder = hashlib.sha256()

    cfg_as_dict = OmegaConf.to_container(cfg)
    get_run_digest(cfg_as_dict, keys_to_ignore, hash_builder)
    run_digest = hash_builder.hexdigest()
    with open_dict(cfg):
        cfg["digest"] = run_digest


def get_run_digest(cfg, keys_to_ignore, hash_builder):

    for key, value in cfg.items():
        if key in keys_to_ignore:
            continue
        elif isinstance(value, Dict):
            get_run_digest(value, keys_to_ignore, hash_builder)
        else:
            key_value = str(key) + str(value)
            hash_builder.update(key_value.encode("utf-8"))


def run(cfg: DictConfig) -> str:
    """Generic train loop.
    Args:
        cfg: run configuration, defined by Hydra in /conf
    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    keys_to_ignore = {
        "seed_index",
        "tags",
        "data_dir",
        "classes_split_path",
        "prototypes_path",
        "best_model_path",
        "storage_dir",
        "colors_path",
        "entity",
        "log_model",
        "job",
    }
    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)
    add_run_digest(cfg, keys_to_ignore)

    seed_index_everything(cfg.train)

    fast_dev_run: bool = cfg.train.trainer.fast_dev_run

    if fast_dev_run:
        handle_fast_dev_run(cfg)

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    G_train, y_train, episodes_test = fetch_dataset(cfg.nn.data)

    accs = []
    for G_test in tqdm(episodes_test):
        supports, y_supports, queries, y_queries = (
            G_test["supports"],
            G_test["y_supports"],
            G_test["queries"],
            G_test["y_queries"],
        )
        for cfg_kernel in cfg.nn.kernels:
            kernel_name = cfg_kernel["_target_"].split(".")[-1]
            kernel = hydra.utils.instantiate(cfg_kernel, _recursive_=False)
            clf = SVC(kernel="linear")

            K_sup = kernel.fit_transform(supports)
            clf.fit(K_sup, y_supports)

            K_qu = kernel.transform(queries)
            y_pred = clf.predict(K_qu)
            acc = accuracy_score(y_queries, y_pred)
            accs.append(acc)
            logger.experiment.log({f"{kernel_name}/accuracy": acc})
        logger.experiment.log({f"{kernel_name}/accuracy_per_epoch": np.array(accs).mean()})

    if logger is not None:
        logger.experiment.finish()

    return logger.run_dir


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="kernel")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":

    main()
