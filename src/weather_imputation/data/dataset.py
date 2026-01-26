"""PyTorch Dataset for time series imputation.

This module provides a PyTorch Dataset class for loading windowed time series
data with masking and station metadata support.
"""

from pathlib import Path
from typing import Literal

import polars as pl
import torch
from torch.utils.data import Dataset

from weather_imputation.config.data import MaskingConfig
from weather_imputation.data.masking import apply_mask
from weather_imputation.data.normalization import Normalizer


class TimeSeriesImputationDataset(Dataset):
    """PyTorch Dataset for time series imputation with windowing.

    Loads time series data and creates sliding windows for training/evaluation.
    Supports synthetic gap generation and station metadata conditioning.

    Args:
        data: Time series data tensor of shape (N, T, V) where:
            - N: number of samples/stations
            - T: time series length
            - V: number of variables
        mask: Boolean mask tensor of shape (N, T, V) where True=observed, False=missing
        timestamps: Timestamps tensor of shape (N, T) with Unix timestamps
        station_features: Optional station metadata tensor of shape (N, F) where
            F is number of features
        window_size: Size of sliding windows in timesteps
        stride: Stride between consecutive windows in timesteps
        masking_strategy: Strategy for synthetic gap generation (None = use existing mask)
        masking_config: Configuration dict for masking (missing_ratio, min_gap_length, etc.)
        apply_synthetic_mask: Whether to apply synthetic masking on top of existing mask

    Example:
        >>> # Create dataset with synthetic masking
        >>> dataset = TimeSeriesImputationDataset(
        ...     data=torch.randn(100, 1000, 6),  # 100 stations, 1000 hours, 6 variables
        ...     mask=torch.ones(100, 1000, 6, dtype=torch.bool),  # All observed initially
        ...     timestamps=torch.arange(100 * 1000).reshape(100, 1000),
        ...     window_size=168,  # 1 week windows
        ...     stride=24,  # 1 day stride
        ...     masking_strategy="mcar",
        ...     masking_config={"missing_ratio": 0.2}
        ... )
        >>> sample = dataset[0]
        >>> sample["observed"].shape  # (168, 6)
    """

    def __init__(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        timestamps: torch.Tensor,
        station_features: torch.Tensor | None = None,
        window_size: int = 168,
        stride: int = 24,
        masking_strategy: Literal["mcar", "mar", "mnar", "realistic"] | None = None,
        masking_config: dict | None = None,
        apply_synthetic_mask: bool = True,
    ) -> None:
        """Initialize TimeSeriesImputationDataset."""
        # Validate shapes
        if data.ndim != 3:
            raise ValueError(f"data must be 3D (N, T, V), got shape {data.shape}")
        if mask.shape != data.shape:
            raise ValueError(
                f"mask shape {mask.shape} must match data shape {data.shape}"
            )
        if timestamps.shape != data.shape[:2]:
            raise ValueError(
                f"timestamps shape {timestamps.shape} must match data shape[:2] {data.shape[:2]}"
            )

        n_samples, seq_length, n_variables = data.shape

        if station_features is not None and station_features.shape[0] != n_samples:
            raise ValueError(
                f"station_features must have {n_samples} samples, "
                f"got {station_features.shape[0]}"
            )

        # Validate window_size and stride first (before using them)
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        if window_size > seq_length:
            raise ValueError(
                f"window_size ({window_size}) cannot exceed sequence length ({seq_length})"
            )

        self.data = data
        self.mask = mask
        self.timestamps = timestamps
        self.station_features = station_features
        self.window_size = window_size
        self.stride = stride
        self.masking_strategy = masking_strategy
        self.masking_config = masking_config or {}
        self.apply_synthetic_mask = apply_synthetic_mask

        self.n_samples = n_samples
        self.seq_length = seq_length
        self.n_variables = n_variables

        # Compute number of windows per sample
        self.windows_per_sample = (
            seq_length - window_size
        ) // stride + 1

        # Total number of windows across all samples
        self.total_windows = self.n_samples * self.windows_per_sample

    def __len__(self) -> int:
        """Return total number of windows in the dataset."""
        return self.total_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single window sample.

        Args:
            idx: Window index (0 to total_windows-1)

        Returns:
            Dictionary containing:
                - observed: Observed values (T, V) with masked positions set to 0
                - mask: Boolean mask (T, V) where True=observed, False=missing
                - target: Ground truth values (T, V) for all positions
                - timestamps: Unix timestamps (T,)
                - station_features: Station metadata (F,) if available
                - sample_idx: Original sample index
                - window_start: Start index of window in original sequence

        Raises:
            IndexError: If idx is out of range
        """
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(
                f"Index {idx} out of range [0, {self.total_windows})"
            )

        # Compute which sample and which window within that sample
        sample_idx = idx // self.windows_per_sample
        window_idx = idx % self.windows_per_sample

        # Compute window start position
        window_start = window_idx * self.stride
        window_end = window_start + self.window_size

        # Extract window from data
        window_data = self.data[sample_idx, window_start:window_end, :]  # (T, V)
        window_mask = self.mask[sample_idx, window_start:window_end, :]  # (T, V)
        window_timestamps = self.timestamps[sample_idx, window_start:window_end]  # (T,)

        # Apply synthetic masking if requested
        if self.apply_synthetic_mask and self.masking_strategy is not None:
            # Add batch dimension for masking function
            window_data_batch = window_data.unsqueeze(0)  # (1, T, V)

            # Extract seed from masking_config (not part of MaskingConfig schema)
            mask_config_params = self.masking_config.copy()
            seed = mask_config_params.pop("seed", None)

            # Create MaskingConfig from parameters
            mask_config = MaskingConfig(
                strategy=self.masking_strategy,
                **mask_config_params,
            )

            # Apply synthetic mask (this creates a new mask, ignoring existing missing values)
            synthetic_mask = apply_mask(
                data=window_data_batch,
                config=mask_config,
                seed=seed,
            )

            # Combine with existing mask (logical AND - value is observed only if both masks say so)
            synthetic_mask = synthetic_mask.squeeze(0)  # (T, V)
            window_mask = window_mask & synthetic_mask

        # Create observed tensor (masked positions set to 0)
        observed = window_data.clone()
        observed[~window_mask] = 0.0

        # Build output dictionary
        output = {
            "observed": observed,
            "mask": window_mask,
            "target": window_data,  # Ground truth
            "timestamps": window_timestamps,
            "sample_idx": torch.tensor(sample_idx, dtype=torch.long),
            "window_start": torch.tensor(window_start, dtype=torch.long),
        }

        # Add station features if available
        if self.station_features is not None:
            output["station_features"] = self.station_features[sample_idx]

        return output


def load_dataset_from_parquet(
    parquet_path: Path,
    variables: list[str],
    window_size: int = 168,
    stride: int = 24,
    masking_strategy: Literal["mcar", "mar", "mnar", "realistic"] | None = None,
    masking_config: dict | None = None,
    normalizer: Normalizer | None = None,
    apply_synthetic_mask: bool = True,
) -> TimeSeriesImputationDataset:
    """Load dataset from preprocessed parquet file.

    Args:
        parquet_path: Path to parquet file containing preprocessed data
        variables: List of variable names to extract
        window_size: Size of sliding windows in timesteps
        stride: Stride between consecutive windows
        masking_strategy: Strategy for synthetic gap generation
        masking_config: Configuration dict for masking
        normalizer: Optional normalizer to apply to data
        apply_synthetic_mask: Whether to apply synthetic masking

    Returns:
        TimeSeriesImputationDataset instance

    Raises:
        FileNotFoundError: If parquet file doesn't exist
        ValueError: If required columns are missing
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    # Load parquet file
    df = pl.read_parquet(parquet_path)

    # Check required columns
    required_cols = ["station_id", "DATE"] + variables
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Group by station and extract data
    # TODO: This is a placeholder - actual implementation will depend on parquet schema
    # For now, raise NotImplementedError
    raise NotImplementedError(
        "load_dataset_from_parquet is not yet implemented. "
        "Use TimeSeriesImputationDataset directly with tensors."
    )
