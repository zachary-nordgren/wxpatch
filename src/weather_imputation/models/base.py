"""Base protocol for imputation models.

This module defines the common interface that all imputation methods must implement,
whether classical (linear, spline, MICE) or deep learning (SAITS, CSDI).
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor
from torch.utils.data import DataLoader


@runtime_checkable
class Imputer(Protocol):
    """Protocol defining the interface for all imputation methods.

    All imputation models (classical and deep learning) must implement this interface
    to ensure consistency across the evaluation framework.

    The protocol uses PyTorch tensors for all data operations, enabling:
    - Consistent interface between classical and neural methods
    - GPU acceleration where applicable
    - Integration with PyTorch training infrastructure
    """

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        """Train the imputation model on the provided data.

        For classical methods (linear, spline, MICE), this may be a no-op
        as they don't require training. For deep learning methods (SAITS, CSDI),
        this performs the full training loop.

        Args:
            train_loader: DataLoader providing training data batches.
                Each batch is a dict with keys:
                - 'observed': (batch, time, vars) tensor with observed values
                - 'mask': (batch, time, vars) boolean tensor (True=observed)
                - 'target': (batch, time, vars) ground truth tensor
            val_loader: Optional DataLoader for validation during training.
                If None, no validation is performed.

        Returns:
            None. The model updates its internal state.

        Raises:
            RuntimeError: If training fails or data format is invalid.
        """
        ...

    def impute(
        self,
        observed: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Impute missing values in the observed data.

        This is the core inference method used during evaluation. Given partially
        observed time series data, it fills in the missing values.

        Args:
            observed: (N, T, V) tensor with observed values.
                Missing positions should be set to 0 or any value (will be ignored).
            mask: (N, T, V) boolean tensor indicating which values are observed.
                True = observed (use this value)
                False = missing (impute this value)

        Returns:
            (N, T, V) tensor with imputed values. Observed positions should remain
            unchanged; missing positions should be filled with predictions.

        Raises:
            RuntimeError: If model is not fitted (for methods requiring training).
            ValueError: If input shapes are inconsistent or invalid.

        Note:
            For deterministic methods, returns a single prediction.
            For probabilistic methods (CSDI), returns the mean of the posterior.
        """
        ...

    def save(self, path: Path) -> None:
        """Save model state to disk.

        For classical methods, this may save configuration/hyperparameters.
        For deep learning methods, this saves model weights and training state.

        Args:
            path: Directory path where model should be saved.
                Will create the directory if it doesn't exist.

        Raises:
            IOError: If saving fails due to permissions or disk space.

        Note:
            Implementations should save all state needed to restore the model,
            including hyperparameters, normalization statistics, and weights.
        """
        ...

    def load(self, path: Path) -> None:
        """Load model state from disk.

        Restores the model to the state it was in when save() was called.

        Args:
            path: Directory path where model was saved.

        Raises:
            FileNotFoundError: If the specified path doesn't exist.
            RuntimeError: If loaded state is incompatible with current model.

        Note:
            After loading, the model should be ready for inference via impute().
        """
        ...


class BaseImputer:
    """Base class providing common functionality for imputation models.

    This is an optional base class that imputation models can inherit from.
    It provides:
    - Common parameter storage
    - Fitted state tracking
    - Default save/load implementation for model metadata

    Models can either:
    1. Inherit from BaseImputer and implement the Imputer protocol
    2. Implement the Imputer protocol directly without inheritance

    Attributes:
        name: Human-readable name of the imputation method.
        _is_fitted: Whether the model has been fitted to training data.
        _hyperparameters: Dictionary storing model hyperparameters.
    """

    def __init__(self, name: str) -> None:
        """Initialize the base imputer.

        Args:
            name: Human-readable name for this imputation method.
        """
        self.name = name
        self._is_fitted = False
        self._hyperparameters: dict = {}

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted to training data.

        Returns:
            True if fit() has been called successfully, False otherwise.
        """
        return self._is_fitted

    def _check_fitted(self) -> None:
        """Check that the model has been fitted.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.name} has not been fitted yet. Call fit() before impute()."
            )

    def _validate_inputs(self, observed: Tensor, mask: Tensor) -> None:
        """Validate input tensor shapes and types.

        Args:
            observed: (N, T, V) tensor with observed values.
            mask: (N, T, V) boolean tensor with observation mask.

        Raises:
            ValueError: If inputs have wrong shape, type, or dimensions.
        """
        if not isinstance(observed, Tensor):
            raise ValueError(
                f"observed must be a torch.Tensor, got {type(observed)}"
            )
        if not isinstance(mask, Tensor):
            raise ValueError(f"mask must be a torch.Tensor, got {type(mask)}")

        if observed.dim() != 3:
            raise ValueError(
                f"observed must be 3D (batch, time, vars), got shape {observed.shape}"
            )
        if mask.dim() != 3:
            raise ValueError(
                f"mask must be 3D (batch, time, vars), got shape {mask.shape}"
            )

        if observed.shape != mask.shape:
            raise ValueError(
                f"observed and mask must have same shape, "
                f"got observed={observed.shape}, mask={mask.shape}"
            )

        if mask.dtype != torch.bool:
            raise ValueError(f"mask must be boolean tensor, got dtype {mask.dtype}")

    def get_hyperparameters(self) -> dict:
        """Get model hyperparameters.

        Returns:
            Dictionary of hyperparameter names and values.
        """
        return self._hyperparameters.copy()

    def _save_metadata(self, path: Path) -> None:
        """Save model metadata (name, hyperparameters, fitted state).

        Args:
            path: Directory where metadata should be saved.
        """
        path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "name": self.name,
            "is_fitted": self._is_fitted,
            "hyperparameters": self._hyperparameters,
        }
        torch.save(metadata, path / "metadata.pt")

    def _load_metadata(self, path: Path) -> dict:
        """Load model metadata from disk.

        Args:
            path: Directory where metadata was saved.

        Returns:
            Dictionary with metadata (name, is_fitted, hyperparameters).

        Raises:
            FileNotFoundError: If metadata file doesn't exist.
        """
        metadata_path = path / "metadata.pt"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        return torch.load(metadata_path, weights_only=False)
