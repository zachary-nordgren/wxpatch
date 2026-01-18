"""Checkpoint management for training runs."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a saved checkpoint."""

    model_name: str
    target_column: str
    epoch: int
    metrics: dict[str, float]
    timestamp: datetime
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "target_column": self.target_column,
            "epoch": self.epoch,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        """Create from dictionary."""
        return cls(
            model_name=data["model_name"],
            target_column=data["target_column"],
            epoch=data["epoch"],
            metrics=data["metrics"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            config=data.get("config", {}),
        )


class CheckpointManager:
    """Manages saving and loading of model checkpoints."""

    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 5):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints

    def save(
        self,
        model: Any,
        metadata: CheckpointMetadata,
        filename: str | None = None,
    ) -> Path:
        """Save a model checkpoint.

        Args:
            model: The model to save
            metadata: Checkpoint metadata
            filename: Optional specific filename

        Returns:
            Path to saved checkpoint
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = metadata.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{metadata.model_name}_{timestamp}.json"

        checkpoint_path = self.checkpoint_dir / filename
        metadata_path = self.checkpoint_dir / f"{filename}.meta.json"

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # TODO: Save actual model state
        # For now, just save the metadata

        logger.info(f"Saved checkpoint to {checkpoint_path}")
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(self, checkpoint_path: Path) -> tuple[Any, CheckpointMetadata]:
        """Load a model checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint

        Returns:
            Tuple of (model, metadata)
        """
        metadata_path = Path(str(checkpoint_path) + ".meta.json")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            metadata = CheckpointMetadata.from_dict(json.load(f))

        # TODO: Load actual model state
        model = None

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return model, metadata

    def list_checkpoints(self) -> list[tuple[Path, CheckpointMetadata]]:
        """List all available checkpoints.

        Returns:
            List of (path, metadata) tuples sorted by timestamp
        """
        if not self.checkpoint_dir.exists():
            return []

        checkpoints = []
        for meta_path in self.checkpoint_dir.glob("*.meta.json"):
            try:
                with open(meta_path) as f:
                    metadata = CheckpointMetadata.from_dict(json.load(f))
                checkpoint_path = Path(str(meta_path).replace(".meta.json", ""))
                checkpoints.append((checkpoint_path, metadata))
            except Exception as e:
                logger.warning(f"Error loading checkpoint metadata {meta_path}: {e}")

        # Sort by timestamp, newest first
        checkpoints.sort(key=lambda x: x[1].timestamp, reverse=True)
        return checkpoints

    def get_best_checkpoint(
        self,
        metric: str = "val_rmse",
        mode: str = "min",
    ) -> tuple[Path, CheckpointMetadata] | None:
        """Get the best checkpoint by a metric.

        Args:
            metric: Metric to compare
            mode: "min" or "max"

        Returns:
            Best checkpoint or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        valid_checkpoints = [
            (path, meta) for path, meta in checkpoints
            if metric in meta.metrics
        ]

        if not valid_checkpoints:
            return None

        if mode == "min":
            return min(valid_checkpoints, key=lambda x: x[1].metrics[metric])
        else:
            return max(valid_checkpoints, key=lambda x: x[1].metrics[metric])

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if we have too many."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self.max_checkpoints:
            return

        # Remove oldest checkpoints
        for path, _ in checkpoints[self.max_checkpoints:]:
            try:
                path.unlink(missing_ok=True)
                Path(str(path) + ".meta.json").unlink(missing_ok=True)
                logger.debug(f"Removed old checkpoint: {path}")
            except Exception as e:
                logger.warning(f"Error removing checkpoint {path}: {e}")
