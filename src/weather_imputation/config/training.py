"""Training configuration classes for weather imputation models.

This module defines Pydantic models for configuring the training process,
including optimizer settings, learning rate scheduling, early stopping, and checkpointing.
"""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from weather_imputation.config.base import BaseConfig


class OptimizerConfig(BaseConfig):
    """Configuration for optimizer.

    Attributes:
        optimizer_type: Type of optimizer (adam, adamw, sgd)
        learning_rate: Initial learning rate
        weight_decay: L2 regularization coefficient
        betas: Adam/AdamW beta coefficients (beta1, beta2)
        momentum: SGD momentum coefficient
        grad_clip_norm: Maximum gradient norm (disabled if None)
        grad_clip_value: Maximum gradient value (disabled if None)
    """

    optimizer_type: Literal["adam", "adamw", "sgd"] = Field(
        default="adamw", description="Optimizer algorithm"
    )
    learning_rate: float = Field(
        default=1e-3, gt=0.0, le=1.0, description="Initial learning rate"
    )
    weight_decay: float = Field(
        default=1e-4, ge=0.0, description="L2 regularization coefficient"
    )
    betas: tuple[float, float] = Field(
        default=(0.9, 0.999), description="Adam/AdamW beta coefficients"
    )
    momentum: float = Field(
        default=0.9, ge=0.0, lt=1.0, description="SGD momentum coefficient"
    )
    grad_clip_norm: float | None = Field(
        default=1.0, gt=0.0, description="Gradient norm clipping threshold (None to disable)"
    )
    grad_clip_value: float | None = Field(
        default=None, gt=0.0, description="Gradient value clipping threshold (None to disable)"
    )

    @field_validator("betas")
    @classmethod
    def validate_betas(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Validate beta coefficients are in valid range."""
        if not (0.0 <= v[0] < 1.0 and 0.0 <= v[1] < 1.0):
            raise ValueError("Beta coefficients must be in range [0, 1)")
        return v


class SchedulerConfig(BaseConfig):
    """Configuration for learning rate scheduler.

    Attributes:
        scheduler_type: Type of scheduler (cosine, plateau, step, onecycle, none)
        warmup_epochs: Number of warmup epochs (linear warmup from 0 to lr)
        patience: Epochs to wait before reducing LR (plateau scheduler)
        factor: LR reduction factor (plateau scheduler)
        step_size: Epochs between LR reductions (step scheduler)
        gamma: LR decay factor (step scheduler)
        min_lr: Minimum learning rate
    """

    scheduler_type: Literal["cosine", "plateau", "step", "onecycle", "none"] = Field(
        default="cosine", description="Learning rate scheduler type"
    )
    warmup_epochs: int = Field(
        default=5, ge=0, description="Number of warmup epochs"
    )
    patience: int = Field(
        default=5, ge=1, description="Patience for plateau scheduler"
    )
    factor: float = Field(
        default=0.5, gt=0.0, lt=1.0, description="LR reduction factor for plateau scheduler"
    )
    step_size: int = Field(
        default=10, ge=1, description="Step size for step scheduler"
    )
    gamma: float = Field(
        default=0.1, gt=0.0, le=1.0, description="Decay factor for step scheduler"
    )
    min_lr: float = Field(
        default=1e-6, gt=0.0, description="Minimum learning rate"
    )


class EarlyStoppingConfig(BaseConfig):
    """Configuration for early stopping.

    Attributes:
        enabled: Whether to enable early stopping
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        monitor: Metric to monitor (val_loss, val_rmse, val_mae)
        mode: Whether lower or higher is better (min or max)
    """

    enabled: bool = Field(default=True, description="Enable early stopping")
    patience: int = Field(
        default=10, ge=1, description="Epochs to wait for improvement"
    )
    min_delta: float = Field(
        default=1e-4, ge=0.0, description="Minimum improvement threshold"
    )
    monitor: Literal["val_loss", "val_rmse", "val_mae", "val_r2"] = Field(
        default="val_loss", description="Metric to monitor"
    )
    mode: Literal["min", "max"] = Field(
        default="min", description="Whether lower (min) or higher (max) is better"
    )


class CheckpointConfig(BaseConfig):
    """Configuration for model checkpointing.

    Attributes:
        save_every_n_epochs: Save checkpoint every N epochs (0 to disable)
        save_every_n_minutes: Save checkpoint every N minutes (0 to disable)
        keep_last_n: Number of recent checkpoints to keep (None = keep all)
        save_best: Whether to save best model checkpoint
        monitor: Metric to monitor for best checkpoint
        mode: Whether lower or higher is better (min or max)
    """

    save_every_n_epochs: int = Field(
        default=1, ge=0, description="Save checkpoint every N epochs (0 to disable)"
    )
    save_every_n_minutes: int = Field(
        default=30, ge=0, description="Save checkpoint every N minutes (0 to disable)"
    )
    keep_last_n: int | None = Field(
        default=3, ge=1, description="Number of recent checkpoints to keep (None = keep all)"
    )
    save_best: bool = Field(
        default=True, description="Save best model checkpoint"
    )
    monitor: Literal["val_loss", "val_rmse", "val_mae", "val_r2"] = Field(
        default="val_loss", description="Metric to monitor for best checkpoint"
    )
    mode: Literal["min", "max"] = Field(
        default="min", description="Whether lower (min) or higher (max) is better"
    )

    @model_validator(mode="after")
    def validate_checkpoint_frequency(self) -> "CheckpointConfig":
        """Validate that at least one checkpoint method is enabled."""
        if self.save_every_n_epochs == 0 and self.save_every_n_minutes == 0 and not self.save_best:
            raise ValueError(
                "At least one checkpoint method must be enabled "
                "(save_every_n_epochs, save_every_n_minutes, or save_best)"
            )
        return self


class TrainingConfig(BaseConfig):
    """Main training configuration.

    Aggregates all training-related configurations including optimizer,
    scheduler, early stopping, and checkpointing.

    Attributes:
        batch_size: Training batch size
        max_epochs: Maximum number of training epochs
        validation_frequency: Validate every N batches (None = once per epoch)
        seed: Random seed for reproducibility
        device: Device to train on (cuda, cpu, or auto)
        mixed_precision: Enable mixed precision training (AMP)
        compile_model: Enable torch.compile for faster training (PyTorch 2.0+)
        num_workers: Number of DataLoader workers
        optimizer: Optimizer configuration
        scheduler: Learning rate scheduler configuration
        early_stopping: Early stopping configuration
        checkpoint: Checkpointing configuration
    """

    batch_size: int = Field(
        default=32, ge=1, description="Training batch size"
    )
    max_epochs: int = Field(
        default=100, ge=1, description="Maximum number of training epochs"
    )
    validation_frequency: int | None = Field(
        default=None, ge=1, description="Validate every N batches (None = once per epoch)"
    )
    seed: int = Field(
        default=42, description="Random seed for reproducibility"
    )
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto", description="Device to train on (auto selects cuda if available)"
    )
    mixed_precision: bool = Field(
        default=True, description="Enable mixed precision training (AMP)"
    )
    compile_model: bool = Field(
        default=False, description="Enable torch.compile for faster training (PyTorch 2.0+)"
    )
    num_workers: int = Field(
        default=4, ge=0, description="Number of DataLoader workers"
    )

    # Nested configurations
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig, description="Optimizer configuration"
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig, description="Learning rate scheduler configuration"
    )
    early_stopping: EarlyStoppingConfig = Field(
        default_factory=EarlyStoppingConfig, description="Early stopping configuration"
    )
    checkpoint: CheckpointConfig = Field(
        default_factory=CheckpointConfig, description="Checkpointing configuration"
    )
