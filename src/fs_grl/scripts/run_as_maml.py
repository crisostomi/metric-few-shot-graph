import logging
from pathlib import Path
from typing import Dict, List

import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO, load_model

import fs_grl  # noqa
from fs_grl.callbacks import build_callbacks, get_checkpoint_callback
from fs_grl.custom_pipelines.as_maml.as_maml import AS_MAML
from fs_grl.utils import handle_fast_dev_run

# Force the execution of __init__.py if this file is executed directly.

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    """Generic train loop.
    Args:
        cfg: run configuration, defined by Hydra in /conf
    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    seed_index_everything(cfg.train)

    fast_dev_run: bool = cfg.train.trainer.fast_dev_run
    if fast_dev_run:
        handle_fast_dev_run(cfg)

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    metadata: Dict = getattr(datamodule, "metadata", None)

    pylogger.info(f"Instantiating <{cfg.nn.model['_target_']}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.nn.model, cfg=cfg.nn.model, _recursive_=False, metadata=metadata
    )

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=cfg.core.storage_dir,
        plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    pylogger.info("Starting meta-training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=template_core.trainer_ckpt_path)

    best_model_path = get_checkpoint_callback(callbacks).best_model_path
    model = load_model(AS_MAML, checkpoint_path=Path(best_model_path + ".zip"))

    hydra.utils.log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule)

    # Mandatory for multi-run
    if logger is not None:
        logger.experiment.finish()

    return logger.run_dir


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="as-maml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
