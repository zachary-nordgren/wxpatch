"""Akima spline interpolation imputation method.

This module implements Akima spline interpolation for imputing missing values
in multivariate time series data. Akima splines are piecewise cubic interpolants
that are stable with respect to outliers and avoid unphysical oscillations.

References:
    Akima, H. (1970). A new method of interpolation and smooth curve fitting based
    on local procedures. Journal of the ACM, 17(4), 589-602.
"""

import json
from pathlib import Path

import torch
from scipy.interpolate import Akima1DInterpolator
from torch import Tensor
from torch.utils.data import DataLoader

from weather_imputation.models.base import BaseImputer


class AkimaSplineImputer(BaseImputer):
    """Impute missing values using Akima spline interpolation.

    This method uses piecewise cubic polynomials (Akima splines) to interpolate
    missing values. Akima splines have several advantages over standard cubic
    splines:
    - Stability with respect to outliers
    - No unphysical oscillations in regions with rapidly changing derivatives
    - Local computation (only uses neighboring points)

    The method operates on PyTorch tensors with shape (N, T, V) where:
    - N is the batch/sample dimension
    - T is the time dimension
    - V is the variable dimension

    Suitable for:
    - Short to medium gaps in smooth, continuous variables
    - Data with outliers that would cause cubic spline artifacts
    - Data with rapidly changing derivatives

    Not suitable for:
    - Very short sequences (<2 observed points)
    - Long gaps (extrapolation at boundaries uses nearest value)
    - Highly variable or noisy data

    Attributes:
        name: "Akima Spline"
        max_gap_length: Maximum gap size to interpolate (in timesteps).
            Gaps longer than this remain missing. If None, all gaps are filled.

    Example:
        >>> imputer = AkimaSplineImputer(max_gap_length=48)
        >>> imputer.fit(train_loader)  # No-op for spline interpolation
        >>> imputed = imputer.impute(observed, mask)

    Note:
        Requires at least 2 observed points for interpolation to be meaningful.
        For sequences with <2 observed points, returns zeros or forwards/backwards
        fills single observed values.
    """

    def __init__(self, max_gap_length: int | None = None):
        """Initialize the Akima spline interpolation imputer.

        Args:
            max_gap_length: Maximum gap size to interpolate (in timesteps).
                If None (default), all gaps are interpolated regardless of size.
                If specified, only gaps of this length or shorter are filled.
        """
        super().__init__(name="Akima Spline")
        self.max_gap_length = max_gap_length
        self._hyperparameters = {"max_gap_length": max_gap_length}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        """Fit the imputer.

        Akima spline interpolation is a non-parametric method that doesn't require
        training. This method just marks the imputer as fitted.

        Args:
            train_loader: Training data (unused for spline interpolation).
            val_loader: Validation data (unused for spline interpolation).
        """
        self._is_fitted = True

    def impute(self, observed: Tensor, mask: Tensor) -> Tensor:
        """Impute missing values using Akima spline interpolation.

        For each sample and variable independently, interpolates missing values
        using Akima splines. Boundary handling:
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
        """Interpolate a single 1D time series using Akima splines.

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

        # If only one value is observed, fill entire series with that value
        n_observed = mask.sum().item()
        if n_observed == 1:
            observed_value = series[mask][0]
            return torch.full_like(series, observed_value)

        result = series.clone()

        # Get indices and values of observed points
        observed_indices = torch.where(mask)[0]
        observed_values = series[observed_indices]

        # Convert to numpy for scipy (Akima1DInterpolator requires numpy)
        x_observed = observed_indices.cpu().numpy()
        y_observed = observed_values.cpu().numpy()

        # Create Akima interpolator
        # Note: Akima requires at least 2 points, which we've ensured above
        try:
            interpolator = Akima1DInterpolator(x_observed, y_observed)
        except ValueError:
            # Fallback to linear if Akima fails (shouldn't happen with >=2 points)
            # This can happen if all observed values are identical
            return self._linear_interpolate_1d(series, mask)

        # Interpolate each gap
        first_observed_idx = observed_indices[0].item()
        last_observed_idx = observed_indices[-1].item()

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

            # Use Akima interpolation for this gap
            gap_indices = torch.arange(
                left_idx + 1, right_idx, device=series.device, dtype=torch.long
            )
            x_gap = gap_indices.cpu().numpy()
            y_gap = interpolator(x_gap)

            # Convert back to torch and fill the gap
            result[gap_indices] = torch.from_numpy(y_gap).to(
                device=series.device, dtype=series.dtype
            )

        # Handle leading missing values (before first observed)
        if first_observed_idx > 0:
            result[:first_observed_idx] = series[first_observed_idx]

        # Handle trailing missing values (after last observed)
        if last_observed_idx < T - 1:
            result[last_observed_idx + 1 :] = series[last_observed_idx]

        return result

    def _linear_interpolate_1d(self, series: Tensor, mask: Tensor) -> Tensor:
        """Fallback to linear interpolation if Akima fails.

        Args:
            series: (T,) tensor with observed values.
            mask: (T,) boolean tensor (True=observed, False=missing).

        Returns:
            (T,) tensor with linearly interpolated values.
        """
        T = series.shape[0]
        result = series.clone()

        observed_indices = torch.where(mask)[0]
        first_observed_idx = observed_indices[0].item()
        last_observed_idx = observed_indices[-1].item()

        # Interpolate each gap linearly
        for i in range(len(observed_indices) - 1):
            left_idx = observed_indices[i].item()
            right_idx = observed_indices[i + 1].item()

            gap_length = right_idx - left_idx - 1
            if gap_length == 0:
                continue

            # Skip if gap is too long
            if self.max_gap_length is not None and gap_length > self.max_gap_length:
                continue

            # Linear interpolation
            left_value = series[left_idx]
            right_value = series[right_idx]

            steps = gap_length + 1
            weights = torch.linspace(
                0, 1, steps + 1, device=series.device, dtype=series.dtype
            )[1:-1]

            interpolated_values = left_value + weights * (right_value - left_value)
            result[left_idx + 1 : right_idx] = interpolated_values

        # Handle boundaries
        if first_observed_idx > 0:
            result[:first_observed_idx] = series[first_observed_idx]
        if last_observed_idx < T - 1:
            result[last_observed_idx + 1 :] = series[last_observed_idx]

        return result

    def save(self, path: Path) -> None:
        """Save model state to disk.

        For Akima spline interpolation, only saves hyperparameters (max_gap_length).

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
