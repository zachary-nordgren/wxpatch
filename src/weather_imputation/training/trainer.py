"""Main training loop for imputation models."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from weather_imputation.models.base import BaseImputer

logger = logging.getLogger(__name__)

# Type alias for callback function
TrainingCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class TrainingConfig:
    """Configuration for training runs."""

    # Data settings
    target_columns: list[str] = field(default_factory=lambda: ["temperature"])
    validation_split: float = 0.2
    test_split: float = 0.1

    # Training settings
    random_seed: int = 42

    # Output settings
    checkpoint_dir: Path = Path("checkpoints")
    log_interval: int = 100

    # Early stopping
    patience: int = 5
    min_delta: float = 0.001


@dataclass
class TrainingResult:
    """Results from a training run."""

    model_name: str
    target_column: str
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float] | None
    training_time_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "target_column": self.target_column,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "training_time_seconds": self.training_time_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


class Trainer:
    """Main trainer class for imputation models."""

    def __init__(
        self,
        config: TrainingConfig | None = None,
        callbacks: list[TrainingCallback] | None = None,
    ):
        """Initialize the trainer.

        Args:
            config: Training configuration
            callbacks: List of callback functions
        """
        self.config = config or TrainingConfig()
        self.callbacks = callbacks or []
        self._results: list[TrainingResult] = []

    def train(
        self,
        model: BaseImputer,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame | None = None,
        target_column: str = "temperature",
    ) -> TrainingResult:
        """Train an imputation model.

        Args:
            model: The imputation model to train
            train_df: Training data
            val_df: Validation data (optional)
            target_column: Column to impute

        Returns:
            TrainingResult with metrics
        """
        start_time = datetime.now()

        logger.info(f"Training {model.name} on column '{target_column}'")

        # Fit the model
        model.fit(train_df, target_column)

        # Evaluate on training data
        train_metrics = self._evaluate(model, train_df, target_column)

        # Evaluate on validation data
        val_metrics = (
            self._evaluate(model, val_df, target_column) if val_df is not None else {}
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        result = TrainingResult(
            model_name=model.name,
            target_column=target_column,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=None,
            training_time_seconds=elapsed,
        )

        self._results.append(result)
        return result

    def _evaluate(
        self,
        model: BaseImputer,
        df: pl.DataFrame,
        target_column: str,
    ) -> dict[str, float]:
        """Evaluate model on a dataset.

        Creates artificial gaps and measures imputation quality.

        Args:
            model: Trained imputation model
            df: Data to evaluate on
            target_column: Column being imputed

        Returns:
            Dictionary of metric names to values
        """
        # TODO: Implement proper evaluation with artificial gap creation
        # For now, return placeholder metrics
        return {
            "rmse": 0.0,
            "mae": 0.0,
            "coverage": 0.0,
        }

    def cross_validate(
        self,
        model: BaseImputer,
        df: pl.DataFrame,
        target_column: str,
        n_folds: int = 5,
    ) -> list[TrainingResult]:
        """Perform k-fold cross-validation.

        Args:
            model: The imputation model to evaluate
            df: Full dataset
            target_column: Column to impute
            n_folds: Number of folds

        Returns:
            List of TrainingResult for each fold
        """
        # TODO: Implement time-series aware cross-validation
        results: list[TrainingResult] = []
        return results

    def get_results(self) -> list[TrainingResult]:
        """Get all training results."""
        return self._results.copy()
