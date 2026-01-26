"""Linear interpolation imputation method.

This module implements time-based linear interpolation for imputing missing values
in multivariate time series data. It fills gaps by drawing straight lines between
observed values.
"""

import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from weather_imputation.models.base import BaseImputer


class LinearInterpolationImputer(BaseImputer):
    """Impute missing values using linear interpolation.

    This method fills missing values by drawing a straight line between
    the nearest non-null values on either side of each gap. Interpolation
    is performed independently for each variable in each sample.

    The method operates on PyTorch tensors with shape (N, T, V) where:
    - N is the batch/sample dimension
    - T is the time dimension
    - V is the variable dimension

    Suitable for:
    - Short gaps in continuous variables (temperature, pressure)
    - Data with gradual, linear changes
    - Quick baseline for comparison

    Not suitable for:
    - Long gaps (extrapolation at boundaries is forward/backward fill)
    - Highly variable or cyclical data (wind direction)
    - Data with non-linear patterns

    Attributes:
        name: "Linear Interpolation"
        max_gap_length: Maximum gap size to interpolate (in timesteps).
            Gaps longer than this remain missing. If None, all gaps are filled.

    Example:
        >>> imputer = LinearInterpolationImputer(max_gap_length=24)
        >>> imputer.fit(train_loader)  # No-op for linear interpolation
        >>> imputed = imputer.impute(observed, mask)
    """

    def __init__(self, max_gap_length: int | None = None):
        """Initialize the linear interpolation imputer.

        Args:
            max_gap_length: Maximum gap size to interpolate (in timesteps).
                If None (default), all gaps are interpolated regardless of size.
                If specified, only gaps of this length or shorter are filled.
        """
        super().__init__(name="Linear Interpolation")
        self.max_gap_length = max_gap_length
        self._hyperparameters = {"max_gap_length": max_gap_length}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        """Fit the imputer.

        Linear interpolation is a non-parametric method that doesn't require
        training. This method just marks the imputer as fitted.

        Args:
            train_loader: Training data (unused for linear interpolation).
            val_loader: Validation data (unused for linear interpolation).
        """
        self._is_fitted = True

    def impute(self, observed: Tensor, mask: Tensor) -> Tensor:
        """Impute missing values using linear interpolation.

        For each sample and variable independently, interpolates missing values
        by drawing straight lines between observed values. Boundary handling:
        - Leading missing values: forward-filled from first observed value
        - Trailing missing values: backward-filled from last observed value

        Args:
            observed: (N, T, V) tensor with observed values.
                Missing positions can contain any value (will be replaced).
            mask: (N, T, V) boolean tensor indicating which values are observed.
                True = observed, False = missing (to be imputed).

        Returns:
            (N, T, V) tensor with imputed values. Observed positions are
            preserved; missing positions are filled with interpolated values.

        Raises:
            RuntimeError: If imputer has not been fitted.
            ValueError: If input tensors have invalid shape or type.
        """
        self._check_fitted()
        self._validate_inputs(observed, mask)

        N, T, V = observed.shape
        result = observed.clone()

        # Process each sample and variable independently
        for n in range(N):
            for v in range(V):
                # Extract single time series
                series = observed[n, :, v]  # (T,)
                series_mask = mask[n, :, v]  # (T,)

                # Interpolate this time series
                interpolated = self._interpolate_1d(series, series_mask)

                # Store result
                result[n, :, v] = interpolated

        return result

    def _interpolate_1d(self, series: Tensor, mask: Tensor) -> Tensor:
        """Interpolate a single 1D time series.

        Args:
            series: (T,) tensor with observed values.
            mask: (T,) boolean tensor (True=observed, False=missing).

        Returns:
            (T,) tensor with interpolated values.
        """
        T = series.shape[0]

        # If all values are observed, no interpolation needed
        if mask.all():
            return series.clone()

        # If no values are observed, return zeros
        if not mask.any():
            return torch.zeros_like(series)

        result = series.clone()

        # Get indices of observed values
        observed_indices = torch.where(mask)[0]

        # Interpolate each gap
        for i in range(len(observed_indices) - 1):
            left_idx = observed_indices[i].item()
            right_idx = observed_indices[i + 1].item()

            # Check if there's a gap
            gap_length = right_idx - left_idx - 1
            if gap_length == 0:
                continue  # No gap

            # Skip if gap is too long (if max_gap_length specified)
            if self.max_gap_length is not None and gap_length > self.max_gap_length:
                continue

            # Linear interpolation
            left_value = series[left_idx]
            right_value = series[right_idx]

            # Generate interpolated values
            steps = gap_length + 1  # Include endpoints
            weights = torch.linspace(
                0, 1, steps + 1, device=series.device, dtype=series.dtype
            )[1:-1]  # Exclude endpoints

            interpolated_values = left_value + weights * (right_value - left_value)

            # Fill the gap
            result[left_idx + 1 : right_idx] = interpolated_values

        # Handle leading missing values (before first observed)
        first_observed_idx = observed_indices[0].item()
        if first_observed_idx > 0:
            result[:first_observed_idx] = series[first_observed_idx]

        # Handle trailing missing values (after last observed)
        last_observed_idx = observed_indices[-1].item()
        if last_observed_idx < T - 1:
            result[last_observed_idx + 1 :] = series[last_observed_idx]

        return result

    def save(self, path: Path) -> None:
        """Save model state to disk.

        For linear interpolation, only saves hyperparameters (max_gap_length).

        Args:
            path: Directory where model should be saved.
        """
        self._save_metadata(path)

        # Save hyperparameters as JSON for human readability
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump(self._hyperparameters, f, indent=2)

    def load(self, path: Path) -> None:
        """Load model state from disk.

        Args:
            path: Directory where model was saved.

        Raises:
            FileNotFoundError: If the specified path doesn't exist.
        """
        metadata = self._load_metadata(path)

        # Restore state
        self._is_fitted = metadata["is_fitted"]
        self._hyperparameters = metadata["hyperparameters"]
        self.max_gap_length = self._hyperparameters.get("max_gap_length")
