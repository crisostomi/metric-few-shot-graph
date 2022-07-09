import hashlib
import logging
from typing import Dict, List

import hydra
import omegaconf
import pytorch_lightning
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Callback

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

import fs_grl  # noqa
from fs_grl.callbacks import build_callbacks
from fs_grl.data.datamodule.datamodule import GraphFewShotDataModule
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

    pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")
    datamodule: GraphFewShotDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    metadata: Dict = getattr(datamodule, "metadata", None)

    pylogger.info(f"Instantiating <{cfg.nn.model['_target_']}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.nn.model, _recursive_=False, train_data_list_by_label=datamodule.data_list_by_base_label, metadata=metadata
    )

    callbacks: List[Callback] = build_callbacks(cfg.train["callbacks"], template_core)

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=cfg.core.storage_dir,
        plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    pylogger.info("Starting training!")
    pylogger.info(f"Digest: {cfg['digest']}")

    trainer.fit(model=model, datamodule=datamodule)

    if fast_dev_run:
        pylogger.info("Skipping testing in 'fast_dev_run' mode!")
    else:
        if trainer.checkpoint_callback.best_model_path is not None:
            pylogger.info("Starting testing!")
            pytorch_lightning.seed_everything(seed=0)
            datamodule: GraphFewShotDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
            datamodule.setup()
            trainer.test(datamodule=datamodule)

    if logger is not None:
        logger.experiment.finish()

    return logger.run_dir


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="mpp")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":

    main()
