from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

from deprecate import void
from pytorch_lightning.loops import EvaluationLoop, FitLoop, TrainingBatchLoop, TrainingEpochLoop
from pytorch_lightning.loops.optimization.manual_loop import _OUTPUTS_TYPE as _MANUAL_LOOP_OUTPUTS_TYPE
from pytorch_lightning.loops.optimization.optimizer_loop import _OUTPUTS_TYPE as _OPTIMIZER_LOOP_OUTPUTS_TYPE
from pytorch_lightning.loops.utilities import _get_active_optimizers

_OUTPUTS_TYPE = List[Union[_OPTIMIZER_LOOP_OUTPUTS_TYPE, _MANUAL_LOOP_OUTPUTS_TYPE]]


class CustomFitLoop(FitLoop):
    def __init__(
        self,
        fine_tuning_epochs: int,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
    ):
        super().__init__()

        min_epochs = 1 if (min_epochs is None and min_steps is None and max_time is None) else min_epochs
        max_epochs = max_epochs if max_epochs is not None else (1000 if (max_steps == -1 and max_time is None) else -1)

        super(CustomFitLoop, self).__init__(min_epochs=min_epochs, max_epochs=max_epochs)

        training_epoch_loop = TrainingEpochLoop(min_steps, max_steps)
        training_batch_loop = CustomTrainingBatchLoop(fine_tuning_epochs)
        training_validation_loop = EvaluationLoop()
        training_epoch_loop.connect(batch_loop=training_batch_loop, val_loop=training_validation_loop)
        self.connect(epoch_loop=training_epoch_loop)


class CustomTrainingBatchLoop(TrainingBatchLoop):
    def __init__(self, fine_tuning_epochs):
        super(CustomTrainingBatchLoop, self).__init__()
        self.fine_tuning_epochs = fine_tuning_epochs

    def advance(self, batch: Any, batch_idx: int) -> None:  # type: ignore[override]
        """Runs the train step together with optimization (if necessary) on the current batch split.

        Args:
            batch: the current batch to run the training on (this is not the split!)
            batch_idx: the index of the current batch
        """
        void(batch)
        self.split_idx, split_batch = self._remaining_splits.pop(0)

        # let logger connector extract current batch size
        self.trainer.logger_connector.on_train_split_start(self.split_idx, split_batch)

        for _ in range(self.fine_tuning_epochs):
            # choose which loop will run the optimization
            if self.trainer.lightning_module.automatic_optimization:
                optimizers = _get_active_optimizers(
                    self.trainer.optimizers, self.trainer.optimizer_frequencies, batch_idx
                )
                outputs = self.optimizer_loop.run(split_batch, optimizers, batch_idx)
            else:
                outputs = self.manual_loop.run(split_batch, batch_idx)
            if outputs:
                # automatic: can be empty if all optimizers skip their batches
                # manual: #9052 added support for raising `StopIteration` in the `training_step`. If that happens,
                # then `advance` doesn't finish and an empty dict is returned
                self._outputs.append(outputs)
