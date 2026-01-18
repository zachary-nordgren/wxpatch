"""Training callbacks for logging and checkpointing."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TrainingCallback(ABC):
    """Base class for training callbacks."""

    @abstractmethod
    def on_train_start(self, trainer: Any) -> None:
        """Called when training starts."""
        ...

    @abstractmethod
    def on_train_end(self, trainer: Any, result: Any) -> None:
        """Called when training ends."""
        ...

    @abstractmethod
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch."""
        ...


class LoggingCallback(TrainingCallback):
    """Callback for logging training progress."""

    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level
        self._start_time: datetime | None = None

    def on_train_start(self, trainer: Any) -> None:
        self._start_time = datetime.now()
        logger.log(self.log_level, "Training started")

    def on_train_end(self, trainer: Any, result: Any) -> None:
        if self._start_time:
            elapsed = datetime.now() - self._start_time
            logger.log(self.log_level, f"Training completed in {elapsed}")

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.log(self.log_level, f"Epoch {epoch}: {metrics_str}")


class CheckpointCallback(TrainingCallback):
    """Callback for saving model checkpoints."""

    def __init__(
        self,
        checkpoint_dir: Path,
        save_best_only: bool = True,
        monitor: str = "val_rmse",
        mode: str = "min",
    ):
        """Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Only save when monitored metric improves
            monitor: Metric to monitor for improvement
            mode: "min" or "max" for the monitored metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self._best_value: float | None = None

    def on_train_start(self, trainer: Any) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._best_value = None

    def on_train_end(self, trainer: Any, result: Any) -> None:
        # Save final checkpoint
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        if self.monitor not in metrics:
            return

        current_value = metrics[self.monitor]

        if self._best_value is None:
            should_save = True
        elif self.mode == "min":
            should_save = current_value < self._best_value
        else:
            should_save = current_value > self._best_value

        if should_save:
            self._best_value = current_value
            self._save_checkpoint(trainer, epoch, metrics)

    def _save_checkpoint(
        self,
        trainer: Any,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Save a checkpoint."""
        # TODO: Implement model serialization
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.json"
        logger.info(f"Saved checkpoint to {checkpoint_path}")


class EarlyStoppingCallback(TrainingCallback):
    """Callback for early stopping."""

    def __init__(
        self,
        monitor: str = "val_rmse",
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min",
    ):
        """Initialize early stopping.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to count as improvement
            mode: "min" or "max"
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best_value: float | None = None
        self._wait: int = 0
        self.stopped_epoch: int = 0
        self.should_stop: bool = False

    def on_train_start(self, trainer: Any) -> None:
        self._best_value = None
        self._wait = 0
        self.should_stop = False

    def on_train_end(self, trainer: Any, result: Any) -> None:
        if self.should_stop:
            logger.info(f"Early stopping at epoch {self.stopped_epoch}")

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        if self.monitor not in metrics:
            return

        current = metrics[self.monitor]

        if self._best_value is None:
            self._best_value = current
            return

        if self.mode == "min":
            improved = current < self._best_value - self.min_delta
        else:
            improved = current > self._best_value + self.min_delta

        if improved:
            self._best_value = current
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch
