"""Variable normalization utilities for weather time series data.

This module implements per-station normalization (z-score and min-max) for
weather variables. Normalization is critical for deep learning models which
require inputs on similar scales.

Implementation details:
- Normalization statistics computed only from observed (non-missing) values
- Statistics stored per-variable and per-station
- Circular variables (wind direction) handled separately (see utils/circular.py)
- Supports both z-score (mean=0, std=1) and min-max (0-1 range) normalization
"""

from typing import Literal

import torch
from pydantic import BaseModel, Field


class NormalizationStats(BaseModel):
    """Statistics for normalizing a single variable.

    Attributes:
        mean: Mean value (for z-score normalization)
        std: Standard deviation (for z-score normalization)
        min_val: Minimum value (for min-max normalization)
        max_val: Maximum value (for min-max normalization)
        n_observed: Number of observed (non-missing) values used to compute stats
    """

    mean: float = Field(description="Mean value for z-score normalization")
    std: float = Field(description="Standard deviation for z-score normalization")
    min_val: float = Field(description="Minimum value for min-max normalization")
    max_val: float = Field(description="Maximum value for min-max normalization")
    n_observed: int = Field(
        description="Number of observed values used to compute statistics"
    )


class Normalizer:
    """Normalizer for weather time series data.

    Computes normalization statistics from observed values and provides
    transform/inverse_transform methods. Handles missing values gracefully.

    Args:
        method: Normalization method ("zscore", "minmax", or "none")
        clip_outliers: Whether to clip outliers beyond 5 std devs (zscore only)

    Example:
        >>> # Normalize a batch of time series
        >>> data = torch.randn(32, 168, 6)  # (batch, time, variables)
        >>> mask = torch.rand(32, 168, 6) > 0.2  # True = observed
        >>> normalizer = Normalizer(method="zscore")
        >>> normalizer.fit(data, mask)
        >>> normalized = normalizer.transform(data, mask)
        >>> reconstructed = normalizer.inverse_transform(normalized)
    """

    def __init__(
        self,
        method: Literal["zscore", "minmax", "none"] = "zscore",
        clip_outliers: bool = False,
    ):
        """Initialize the normalizer.

        Args:
            method: Normalization method
            clip_outliers: Whether to clip outliers beyond 5 std devs (zscore only)
        """
        self.method = method
        self.clip_outliers = clip_outliers
        self.stats: dict[int, NormalizationStats] | None = None
        self._fitted = False

    def fit(self, data: torch.Tensor, mask: torch.Tensor) -> None:
        """Compute normalization statistics from observed values.

        Args:
            data: Time series data (N, T, V) where N=samples, T=timesteps, V=variables
            mask: Boolean mask (N, T, V) where True=observed, False=missing

        Raises:
            ValueError: If data and mask shapes don't match
            ValueError: If no observed values for a variable
        """
        if data.shape != mask.shape:
            raise ValueError(
                f"Data shape {data.shape} must match mask shape {mask.shape}"
            )

        if data.ndim != 3:
            raise ValueError(
                f"Data must be 3D (N, T, V), got shape {data.shape} ({data.ndim}D)"
            )

        _, _, n_vars = data.shape

        # Compute stats per variable
        self.stats = {}
        for v in range(n_vars):
            # Extract observed values for this variable
            var_mask = mask[:, :, v]
            var_data = data[:, :, v]
            observed = var_data[var_mask]

            if len(observed) == 0:
                raise ValueError(
                    f"Variable {v} has no observed values - cannot compute normalization stats"
                )

            # Compute statistics
            mean = float(observed.mean())
            std = float(observed.std())
            min_val = float(observed.min())
            max_val = float(observed.max())
            n_observed = len(observed)

            # Handle edge case: constant variable (std = 0)
            if std == 0.0:
                std = 1.0  # Avoid division by zero

            # Handle edge case: constant variable (min = max)
            if min_val == max_val:
                max_val = min_val + 1.0  # Avoid division by zero

            self.stats[v] = NormalizationStats(
                mean=mean,
                std=std,
                min_val=min_val,
                max_val=max_val,
                n_observed=n_observed,
            )

        self._fitted = True

    def transform(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Normalize the data using computed statistics.

        Only observed values (mask=True) are normalized. Missing values are
        left unchanged (they will be filled by imputation later).

        Args:
            data: Time series data (N, T, V)
            mask: Boolean mask (N, T, V) where True=observed, False=missing

        Returns:
            Normalized data (N, T, V) with same shape as input

        Raises:
            ValueError: If normalizer not fitted
            ValueError: If data shape doesn't match fitted data
        """
        if not self._fitted:
            raise ValueError("Normalizer must be fitted before transform()")

        if data.ndim != 3:
            raise ValueError(
                f"Data must be 3D (N, T, V), got shape {data.shape} ({data.ndim}D)"
            )

        if data.shape[2] != len(self.stats):
            raise ValueError(
                f"Data has {data.shape[2]} variables but normalizer was "
                f"fitted with {len(self.stats)} variables"
            )

        if self.method == "none":
            return data.clone()

        # Clone data to avoid modifying input
        normalized = data.clone()

        # Normalize each variable
        for v in range(data.shape[2]):
            var_mask = mask[:, :, v]
            var_data = normalized[:, :, v]
            stats = self.stats[v]

            if self.method == "zscore":
                # Z-score normalization: (x - mean) / std
                var_data[var_mask] = (var_data[var_mask] - stats.mean) / stats.std

                # Clip outliers if requested
                if self.clip_outliers:
                    var_data[var_mask] = torch.clamp(
                        var_data[var_mask], min=-5.0, max=5.0
                    )

            elif self.method == "minmax":
                # Min-max normalization: (x - min) / (max - min)
                var_data[var_mask] = (var_data[var_mask] - stats.min_val) / (
                    stats.max_val - stats.min_val
                )

            # Update the variable in the normalized tensor
            normalized[:, :, v] = var_data

        return normalized

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize the data back to original scale.

        Args:
            data: Normalized time series data (N, T, V)

        Returns:
            Denormalized data (N, T, V) with same shape as input

        Raises:
            ValueError: If normalizer not fitted
            ValueError: If data shape doesn't match fitted data
        """
        if not self._fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform()")

        if data.ndim != 3:
            raise ValueError(
                f"Data must be 3D (N, T, V), got shape {data.shape} ({data.ndim}D)"
            )

        if data.shape[2] != len(self.stats):
            raise ValueError(
                f"Data has {data.shape[2]} variables but normalizer was "
                f"fitted with {len(self.stats)} variables"
            )

        if self.method == "none":
            return data.clone()

        # Clone data to avoid modifying input
        denormalized = data.clone()

        # Denormalize each variable
        for v in range(data.shape[2]):
            var_data = denormalized[:, :, v]
            stats = self.stats[v]

            if self.method == "zscore":
                # Inverse z-score: x_orig = (x_norm * std) + mean
                var_data = (var_data * stats.std) + stats.mean

            elif self.method == "minmax":
                # Inverse min-max: x_orig = (x_norm * (max - min)) + min
                var_data = (var_data * (stats.max_val - stats.min_val)) + stats.min_val

            denormalized[:, :, v] = var_data

        return denormalized

    def get_stats(self, variable_idx: int) -> NormalizationStats | None:
        """Get normalization statistics for a specific variable.

        Args:
            variable_idx: Variable index (0-indexed)

        Returns:
            NormalizationStats for the variable, or None if not fitted
        """
        if not self._fitted:
            return None
        return self.stats.get(variable_idx)

    def is_fitted(self) -> bool:
        """Check if normalizer has been fitted.

        Returns:
            True if fitted, False otherwise
        """
        return self._fitted


def normalize_variables(
    data: torch.Tensor,
    mask: torch.Tensor,
    method: Literal["zscore", "minmax", "none"] = "zscore",
    clip_outliers: bool = False,
) -> tuple[torch.Tensor, Normalizer]:
    """Convenience function to fit and transform in one step.

    Args:
        data: Time series data (N, T, V)
        mask: Boolean mask (N, T, V) where True=observed, False=missing
        method: Normalization method
        clip_outliers: Whether to clip outliers beyond 5 std devs (zscore only)

    Returns:
        Tuple of (normalized_data, fitted_normalizer)

    Example:
        >>> data = torch.randn(32, 168, 6)
        >>> mask = torch.rand(32, 168, 6) > 0.2
        >>> normalized, normalizer = normalize_variables(data, mask, method="zscore")
        >>> # Later: denormalize predictions
        >>> predictions = model(normalized)
        >>> denormalized = normalizer.inverse_transform(predictions)
    """
    normalizer = Normalizer(method=method, clip_outliers=clip_outliers)
    normalizer.fit(data, mask)
    normalized = normalizer.transform(data, mask)
    return normalized, normalizer
