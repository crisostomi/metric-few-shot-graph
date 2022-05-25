import logging
from pathlib import Path
from typing import Dict, List

import hydra
import omegaconf
import pytorch_lightning
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
from fs_grl.data.datamodule import GraphFewShotDataModule

# Force the execution of __init__.py if this file is executed directly.
from fs_grl.modules.meta_learning_loop import CustomFitLoop
from fs_grl.pl_modules.gsm_source import GraphSpectralMeasuresSource

pylogger = logging.getLogger(__name__)


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

    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    seed_index_everything(cfg.train)

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")
    datamodule: GraphFewShotDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    metadata: Dict = getattr(datamodule, "metadata", None)

    pylogger.info(f"Instantiating <{cfg.nn.model.source['_target_']}>")

    model: pl.LightningModule = hydra.utils.instantiate(cfg.nn.model.source, _recursive_=False, metadata=metadata)

    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

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

    pylogger.info("Starting meta-testing.")

    best_model_path = get_checkpoint_callback(callbacks).best_model_path
    best_model = load_model(GraphSpectralMeasuresSource, checkpoint_path=Path(best_model_path + ".zip"))

    callbacks: List[Callback] = build_callbacks(cfg.train["meta-testing-callbacks"], template_core)

    # Create a new NNLogger but set it to resume the just closed one
    resume_id: str = logger.experiment.id

    if logger is not None:
        logger.experiment.finish()

    logger: NNLogger = NNLogger(logging_cfg=cfg.train["meta-testing-logging"], cfg=cfg, resume_id=resume_id)

    trainer = pl.Trainer(
        default_root_dir=cfg.core.storage_dir,
        logger=logger,
        callbacks=callbacks,
        **cfg.train["meta-testing-trainer"],
        checkpoint_callback=False,
    )

    target_model: pl.LightningModule = hydra.utils.instantiate(
        cfg.nn.model.target, _recursive_=False, metadata=metadata, embedder=best_model.embedder
    )

    # Test with a fixed seed for reproducibility
    pytorch_lightning.seed_everything(seed=0)
    datamodule: GraphFewShotDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    datamodule.setup()

    trainer.fit_loop = CustomFitLoop(
        fine_tuning_steps=cfg.train.fine_tuning_steps, max_epochs=cfg.train["meta-testing-trainer"].max_epochs
    )

    trainer.fit(model=target_model, train_dataloader=datamodule.test_dataloader()[0])

    if logger is not None:
        logger.experiment.finish()

    return logger.run_dir


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":

    main()
