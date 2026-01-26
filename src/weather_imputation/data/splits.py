"""Train/validation/test splitting strategies for time series data.

This module implements different strategies for splitting weather station data
into training, validation, and test sets for imputation model development.

Strategy D (simulated masks) is the default and preferred approach: it uses all
stations and simulates realistic missing patterns within the signals themselves,
rather than splitting by station or time.
"""

import random
from typing import Literal

import polars as pl
import torch


def split_spatial(
    metadata: pl.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Split stations spatially into train/val/test sets.

    Randomly assigns complete stations to train, validation, or test sets.
    Ensures geographic diversity across splits but may not capture temporal patterns.

    Args:
        metadata: DataFrame with station_id column
        train_ratio: Proportion of stations for training (0.0-1.0)
        val_ratio: Proportion of stations for validation (0.0-1.0)
        test_ratio: Proportion of stations for testing (0.0-1.0)
        seed: Random seed for reproducibility (None = non-deterministic)

    Returns:
        Tuple of (train_ids, val_ids, test_ids) where each is a list of station IDs

    Raises:
        ValueError: If ratios don't sum to 1.0 or metadata is empty

    Example:
        >>> metadata = pl.DataFrame({"station_id": ["A", "B", "C", "D"]})
        >>> train, val, test = split_spatial(metadata, seed=42)
        >>> len(train) + len(val) + len(test) == 4
        True
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    if len(metadata) == 0:
        raise ValueError("Metadata DataFrame is empty")

    # Get all station IDs
    station_ids = metadata["station_id"].to_list()

    # Shuffle stations
    if seed is not None:
        random.seed(seed)
    random.shuffle(station_ids)

    # Compute split indices
    n_total = len(station_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Split
    train_ids = station_ids[:n_train]
    val_ids = station_ids[n_train : n_train + n_val]
    test_ids = station_ids[n_train + n_val :]

    return train_ids, val_ids, test_ids


def split_temporal(
    metadata: pl.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Split stations temporally into train/val/test sets.

    Assigns earliest data to training, middle data to validation, and latest data
    to testing. Useful for evaluating extrapolation to future time periods.

    Note: This strategy requires temporal metadata (first_observation, last_observation).
    All stations are included in all splits, but with different time ranges.

    Args:
        metadata: DataFrame with station_id, first_observation, last_observation columns
        train_ratio: Proportion of time range for training (0.0-1.0)
        val_ratio: Proportion of time range for validation (0.0-1.0)
        test_ratio: Proportion of time range for testing (0.0-1.0)
        seed: Random seed (unused, kept for consistency)

    Returns:
        Tuple of (train_ids, val_ids, test_ids) where each contains all station IDs
        (temporal filtering happens at data loading time)

    Raises:
        ValueError: If ratios don't sum to 1.0 or required columns missing

    Example:
        >>> metadata = pl.DataFrame({
        ...     "station_id": ["A", "B"],
        ...     "first_observation": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
        ...     "last_observation": [datetime(2023, 1, 1), datetime(2023, 1, 1)]
        ... })
        >>> train, val, test = split_temporal(metadata)
        >>> train == val == test  # All stations in all splits
        True
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    required_cols = {"station_id", "first_observation", "last_observation"}
    missing_cols = required_cols - set(metadata.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # For temporal split, all stations go into all splits
    # Temporal filtering happens during data loading based on observation dates
    station_ids = metadata["station_id"].to_list()

    return station_ids, station_ids, station_ids


def split_hybrid(
    metadata: pl.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    spatial_weight: float = 0.5,
    seed: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Hybrid spatial-temporal split.

    Combines spatial and temporal splitting strategies. A portion of stations
    are held out spatially, while the remaining stations use temporal splitting.

    Args:
        metadata: DataFrame with station metadata
        train_ratio: Proportion for training (0.0-1.0)
        val_ratio: Proportion for validation (0.0-1.0)
        test_ratio: Proportion for testing (0.0-1.0)
        spatial_weight: Weight of spatial vs temporal split (0.0=all temporal, 1.0=all spatial)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_ids, val_ids, test_ids)

    Raises:
        ValueError: If ratios don't sum to 1.0 or spatial_weight out of range

    Example:
        >>> metadata = pl.DataFrame({"station_id": ["A", "B", "C", "D"]})
        >>> train, val, test = split_hybrid(metadata, spatial_weight=0.5, seed=42)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    if not 0.0 <= spatial_weight <= 1.0:
        raise ValueError(f"spatial_weight must be in [0.0, 1.0], got {spatial_weight}")

    # First do spatial split to hold out some stations
    n_total = len(metadata)
    n_spatial_holdout = int(n_total * test_ratio * spatial_weight)

    station_ids = metadata["station_id"].to_list()
    if seed is not None:
        random.seed(seed)
    random.shuffle(station_ids)

    # Stations held out spatially (test only)
    spatial_test_ids = station_ids[:n_spatial_holdout]

    # Remaining stations use temporal split
    temporal_stations = station_ids[n_spatial_holdout:]

    # Adjust ratios for temporal split (redistribute test portion)
    adjusted_train = train_ratio / (train_ratio + val_ratio + test_ratio * (1 - spatial_weight))
    adjusted_val = val_ratio / (train_ratio + val_ratio + test_ratio * (1 - spatial_weight))

    n_temporal = len(temporal_stations)
    n_train = int(n_temporal * adjusted_train)
    n_val = int(n_temporal * adjusted_val)

    train_ids = temporal_stations[:n_train]
    val_ids = temporal_stations[n_train : n_train + n_val]
    test_ids = temporal_stations[n_train + n_val :] + spatial_test_ids

    return train_ids, val_ids, test_ids


def split_simulated(
    metadata: pl.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Strategy D: Simulated masks within signals (preferred for imputation research).

    All stations are included in all splits. Missing patterns are simulated within
    each station's time series using masking strategies (MCAR/MAR/MNAR/realistic).
    This approach:
    - Maximizes data utilization (all stations contribute to all splits)
    - Enables controlled evaluation of imputation under known missingness mechanisms
    - Avoids spatial/temporal distribution shift issues
    - Matches SAITS/CSDI training methodology from research papers

    The split ratios control how much of each station's data goes into train/val/test
    by time, but in practice all data is used with different synthetic masks applied.

    Implementation Note: Actual masking happens in data.masking module during
    Dataset creation. This function simply returns all stations for all splits,
    signaling that masking-based splits should be used.

    Args:
        metadata: DataFrame with station_id column
        train_ratio: Proportion for training (used in masking strategy)
        val_ratio: Proportion for validation (used in masking strategy)
        test_ratio: Proportion for testing (used in masking strategy)
        seed: Random seed for reproducibility (passed to masking functions)

    Returns:
        Tuple of (train_ids, val_ids, test_ids) where all three contain all station IDs.
        The actual train/val/test distinction is enforced by applying different
        synthetic masks during Dataset creation.

    Example:
        >>> metadata = pl.DataFrame({"station_id": ["A", "B", "C"]})
        >>> train, val, test = split_simulated(metadata)
        >>> train == val == test == ["A", "B", "C"]
        True
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    if len(metadata) == 0:
        raise ValueError("Metadata DataFrame is empty")

    # Strategy D: All stations in all splits
    # Differentiation happens via synthetic masking during training
    station_ids = metadata["station_id"].to_list()

    return station_ids, station_ids, station_ids


def create_split(
    metadata: pl.DataFrame,
    strategy: Literal["spatial", "temporal", "hybrid", "simulated"] = "simulated",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int | None = None,
    **kwargs,
) -> tuple[list[str], list[str], list[str]]:
    """Create train/val/test split using specified strategy.

    Dispatcher function that routes to the appropriate splitting strategy.

    Args:
        metadata: DataFrame containing station metadata (must have station_id column)
        strategy: Splitting strategy to use
        train_ratio: Proportion of data for training (0.0-1.0)
        val_ratio: Proportion of data for validation (0.0-1.0)
        test_ratio: Proportion of data for testing (0.0-1.0)
        seed: Random seed for reproducibility (None = non-deterministic)
        **kwargs: Additional strategy-specific parameters (e.g., spatial_weight for hybrid)

    Returns:
        Tuple of (train_ids, val_ids, test_ids) where each is a list of station IDs

    Raises:
        ValueError: If strategy is unknown or ratios invalid

    Example:
        >>> metadata = pl.DataFrame({"station_id": ["A", "B", "C", "D"]})
        >>> train, val, test = create_split(metadata, strategy="spatial", seed=42)
        >>> len(train) >= len(val)  # Training set should be largest
        True
    """
    strategies = {
        "spatial": split_spatial,
        "temporal": split_temporal,
        "hybrid": split_hybrid,
        "simulated": split_simulated,
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Must be one of: {list(strategies.keys())}"
        )

    split_fn = strategies[strategy]

    # Pass common args
    common_args = {
        "metadata": metadata,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
    }

    # Add strategy-specific kwargs
    if strategy == "hybrid":
        common_args["spatial_weight"] = kwargs.get("spatial_weight", 0.5)

    return split_fn(**common_args)


def create_temporal_mask(
    n_timesteps: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> torch.Tensor:
    """Create temporal mask for Strategy D simulated splits.

    Divides a time series into train/val/test segments by time. This mask indicates
    which portion of a time series belongs to which split when using simulated
    masking strategy.

    Args:
        n_timesteps: Total number of timesteps in the time series
        train_ratio: Proportion for training (0.0-1.0)
        val_ratio: Proportion for validation (0.0-1.0)
        test_ratio: Proportion for testing (0.0-1.0)

    Returns:
        Integer tensor of shape (n_timesteps,) where:
        - 0 = training timesteps
        - 1 = validation timesteps
        - 2 = test timesteps

    Raises:
        ValueError: If ratios don't sum to 1.0 or n_timesteps < 1

    Example:
        >>> mask = create_temporal_mask(100, 0.7, 0.15, 0.15)
        >>> mask.shape
        torch.Size([100])
        >>> (mask == 0).sum().item()  # Should be ~70 training timesteps
        70
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    if n_timesteps < 1:
        raise ValueError(f"n_timesteps must be >= 1, got {n_timesteps}")

    # Compute split indices
    n_train = int(n_timesteps * train_ratio)
    n_val = int(n_timesteps * val_ratio)

    # Create mask
    mask = torch.zeros(n_timesteps, dtype=torch.long)
    mask[n_train : n_train + n_val] = 1  # Validation
    mask[n_train + n_val :] = 2  # Test

    return mask


def split_by_temporal_mask(
    data: torch.Tensor,
    mask: torch.Tensor,
    split: Literal["train", "val", "test"],
) -> torch.Tensor:
    """Extract train/val/test portion from data using temporal mask.

    Args:
        data: Data tensor of shape (T, V) or (N, T, V)
        mask: Temporal mask from create_temporal_mask() of shape (T,)
        split: Which split to extract ("train", "val", or "test")

    Returns:
        Data tensor with only the requested split timesteps

    Raises:
        ValueError: If split is unknown or shapes incompatible

    Example:
        >>> data = torch.randn(100, 6)  # 100 timesteps, 6 variables
        >>> mask = create_temporal_mask(100, 0.7, 0.15, 0.15)
        >>> train_data = split_by_temporal_mask(data, mask, "train")
        >>> train_data.shape[0] == 70  # ~70% of timesteps
        True
    """
    split_map = {"train": 0, "val": 1, "test": 2}

    if split not in split_map:
        raise ValueError(f"Unknown split '{split}'. Must be one of: {list(split_map.keys())}")

    split_idx = split_map[split]

    # Handle both 2D (T, V) and 3D (N, T, V) data
    if data.ndim == 2:
        return data[mask == split_idx]
    elif data.ndim == 3:
        return data[:, mask == split_idx]
    else:
        raise ValueError(f"Data must be 2D or 3D, got shape {data.shape}")
