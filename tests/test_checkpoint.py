from importlib import import_module
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule

from tests.conftest import load_checkpoint

from fs_grl.pl_modules.distance_metric_learning import DistanceMetricLearning
from fs_grl.scripts.run_dml import run


def test_load_checkpoint(run_trainings_not_dry: str, cfg_all_not_dry: DictConfig) -> None:
    ckpts_path = Path(run_trainings_not_dry) / "checkpoints"
    checkpoint_path = next(ckpts_path.glob("*"))
    assert checkpoint_path

    reference: str = cfg_all_not_dry.nn.module._target_
    module_ref, class_ref = reference.rsplit(".", maxsplit=1)
    module_class: LightningModule = getattr(import_module(module_ref), class_ref)
    assert module_class is not None

    module = module_class.load_from_checkpoint(checkpoint_path=checkpoint_path)
    assert module is not None
    assert sum(p.numel() for p in module.parameters())


def _check_cfg_in_checkpoint(checkpoint: Dict, _cfg: DictConfig) -> Dict:
    assert "cfg" in checkpoint
    assert checkpoint["cfg"] == _cfg


def _check_run_path_in_checkpoint(checkpoint: Dict) -> Dict:
    assert "run_path" in checkpoint
    assert checkpoint["run_path"]
    checkpoint["run_path"]: str
    assert checkpoint["run_path"].startswith("//")


def test_cfg_in_checkpoint(run_trainings_not_dry: str, cfg_all_not_dry: DictConfig) -> None:
    checkpoint = load_checkpoint(run_trainings_not_dry)

    _check_cfg_in_checkpoint(checkpoint, cfg_all_not_dry)
    _check_run_path_in_checkpoint(checkpoint)


class ModuleWithCustomCheckpoint(DistanceMetricLearning):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["test_key"] = "test_value"


def test_on_save_checkpoint_hook(cfg_all_not_dry: DictConfig) -> None:
    cfg = OmegaConf.create(cfg_all_not_dry)
    cfg.nn.module._target_ = "tests.test_checkpoint.ModuleWithCustomCheckpoint"
    output_path = Path(run(cfg))

    checkpoint = load_checkpoint(output_path)

    _check_cfg_in_checkpoint(checkpoint, cfg)
    _check_run_path_in_checkpoint(checkpoint)

    assert "test_key" in checkpoint
    assert checkpoint["test_key"] == "test_value"
