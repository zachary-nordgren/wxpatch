"""Stratified evaluation by gap length, season, variable, and extremes.

This module provides stratification utilities for analyzing imputation performance
across different data characteristics using PyTorch tensors.

All functions follow conventions from metrics.py:
- Input tensors have shape (N, T, V) where N=samples, T=timesteps, V=variables
- mask tensor: True=evaluate, False=ignore (typically synthetic gaps)
- Returns dictionaries with metrics per stratum
"""

import torch
from torch import Tensor

from weather_imputation.evaluation.metrics import compute_all_metrics


def stratify_by_gap_length(
    y_true: Tensor,
    y_pred: Tensor,
    mask: Tensor,
    gap_lengths: Tensor,
    bins: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute metrics stratified by gap length.

    Groups evaluation positions by the length of the gap they belong to,
    then computes metrics separately for each gap length bin.

    Args:
        y_true: Ground truth values, shape (N, T, V)
        y_pred: Predicted values, shape (N, T, V)
        mask: Evaluation mask, shape (N, T, V), True=evaluate
        gap_lengths: Gap length in hours for each position, shape (N, T, V)
        bins: Gap length bins in hours (e.g., [6, 24, 72, 168])
              Defaults to [6, 24, 72, 168] (6h, 1d, 3d, 1w)

    Returns:
        Dictionary mapping stratum label to metrics dict.
        Keys like "1-6h", "6-24h", "24-72h", "72-168h", ">168h"

    Example:
        >>> y_true = torch.randn(10, 100, 6)  # 10 samples, 100 timesteps, 6 vars
        >>> y_pred = y_true + torch.randn(10, 100, 6) * 0.1
        >>> mask = torch.rand(10, 100, 6) < 0.3  # 30% evaluation positions
        >>> gap_lengths = torch.randint(1, 200, (10, 100, 6)).float()
        >>> results = stratify_by_gap_length(y_true, y_pred, mask, gap_lengths)
        >>> print(results.keys())  # ["1-6h", "6-24h", ...]
    """
    if bins is None:
        bins = [6, 24, 72, 168]

    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, "
            f"mask {mask.shape}"
        )

    if gap_lengths.shape != mask.shape:
        raise ValueError(
            f"gap_lengths shape {gap_lengths.shape} must match mask {mask.shape}"
        )

    results = {}

    # First bin: 1 to bins[0]
    lower = 1
    upper = bins[0]
    bin_mask = mask & (gap_lengths >= lower) & (gap_lengths < upper)
    if bin_mask.any():
        metrics = compute_all_metrics(y_true, y_pred, bin_mask)
        results[f"{lower}-{upper}h"] = metrics

    # Middle bins
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        bin_mask = mask & (gap_lengths >= lower) & (gap_lengths < upper)
        if bin_mask.any():
            metrics = compute_all_metrics(y_true, y_pred, bin_mask)
            results[f"{lower}-{upper}h"] = metrics

    # Final bin: >bins[-1]
    lower = bins[-1]
    bin_mask = mask & (gap_lengths >= lower)
    if bin_mask.any():
        metrics = compute_all_metrics(y_true, y_pred, bin_mask)
        results[f">{lower}h"] = metrics

    return results


def stratify_by_variable(
    y_true: Tensor,
    y_pred: Tensor,
    mask: Tensor,
    variable_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute metrics stratified by variable (e.g., temperature, pressure).

    Evaluates imputation performance separately for each variable in the tensor.

    Args:
        y_true: Ground truth values, shape (N, T, V)
        y_pred: Predicted values, shape (N, T, V)
        mask: Evaluation mask, shape (N, T, V), True=evaluate
        variable_names: Names for each variable dimension (length V)
                       Defaults to ["var_0", "var_1", ..., "var_V-1"]

    Returns:
        Dictionary mapping variable name to metrics dict.

    Example:
        >>> y_true = torch.randn(10, 100, 3)
        >>> y_pred = y_true + torch.randn(10, 100, 3) * 0.1
        >>> mask = torch.rand(10, 100, 3) < 0.3
        >>> names = ["temperature", "pressure", "humidity"]
        >>> results = stratify_by_variable(y_true, y_pred, mask, names)
        >>> print(results.keys())  # ["temperature", "pressure", "humidity"]
    """
    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, "
            f"mask {mask.shape}"
        )

    n_variables = y_true.shape[2]

    if variable_names is None:
        variable_names = [f"var_{i}" for i in range(n_variables)]

    if len(variable_names) != n_variables:
        raise ValueError(
            f"variable_names length {len(variable_names)} must match "
            f"number of variables {n_variables}"
        )

    results = {}

    for var_idx, var_name in enumerate(variable_names):
        # Create mask that selects only this variable
        var_mask = torch.zeros_like(mask, dtype=torch.bool)
        var_mask[:, :, var_idx] = mask[:, :, var_idx]

        if var_mask.any():
            metrics = compute_all_metrics(y_true, y_pred, var_mask)
            results[var_name] = metrics

    return results


def stratify_by_extreme_values(
    y_true: Tensor,
    y_pred: Tensor,
    mask: Tensor,
    percentiles: tuple[float, float] = (5.0, 95.0),
) -> dict[str, dict[str, float]]:
    """Compute metrics stratified by extreme vs normal values.

    Splits evaluation positions into three groups based on ground truth percentiles:
    - "extreme_low": Below lower percentile threshold
    - "normal": Between percentiles
    - "extreme_high": Above upper percentile threshold

    This is useful for understanding if imputation quality degrades at extreme
    weather conditions (heat waves, cold snaps, storms).

    Args:
        y_true: Ground truth values, shape (N, T, V)
        y_pred: Predicted values, shape (N, T, V)
        mask: Evaluation mask, shape (N, T, V), True=evaluate
        percentiles: (lower, upper) percentile thresholds (0-100)
                    Defaults to (5, 95) for 5th and 95th percentiles

    Returns:
        Dictionary with keys "extreme_low", "normal", "extreme_high"
        mapping to metrics dicts.

    Example:
        >>> y_true = torch.randn(10, 100, 6)
        >>> y_pred = y_true + torch.randn(10, 100, 6) * 0.1
        >>> mask = torch.rand(10, 100, 6) < 0.3
        >>> results = stratify_by_extreme_values(y_true, y_pred, mask)
        >>> print(results.keys())  # ["extreme_low", "normal", "extreme_high"]
    """
    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, "
            f"mask {mask.shape}"
        )

    if not (0 <= percentiles[0] < percentiles[1] <= 100):
        raise ValueError(
            f"percentiles must satisfy 0 <= lower < upper <= 100, "
            f"got {percentiles}"
        )

    # Compute percentiles from all ground truth values
    y_true_flat = y_true[mask]

    if y_true_flat.numel() == 0:
        return {}

    # Use quantile instead of percentile (quantile expects 0-1 range)
    lower_threshold = torch.quantile(y_true_flat, percentiles[0] / 100.0)
    upper_threshold = torch.quantile(y_true_flat, percentiles[1] / 100.0)

    results = {}

    # Extreme low: values below lower percentile
    extreme_low_mask = mask & (y_true < lower_threshold)
    if extreme_low_mask.any():
        metrics = compute_all_metrics(y_true, y_pred, extreme_low_mask)
        results["extreme_low"] = metrics

    # Normal: values between percentiles
    normal_mask = mask & (y_true >= lower_threshold) & (y_true <= upper_threshold)
    if normal_mask.any():
        metrics = compute_all_metrics(y_true, y_pred, normal_mask)
        results["normal"] = metrics

    # Extreme high: values above upper percentile
    extreme_high_mask = mask & (y_true > upper_threshold)
    if extreme_high_mask.any():
        metrics = compute_all_metrics(y_true, y_pred, extreme_high_mask)
        results["extreme_high"] = metrics

    return results


def stratify_by_season(
    y_true: Tensor,
    y_pred: Tensor,
    mask: Tensor,
    timestamps: Tensor,
) -> dict[str, dict[str, float]]:
    """Compute metrics stratified by meteorological season.

    Seasons defined as:
    - winter: December, January, February
    - spring: March, April, May
    - summer: June, July, August
    - fall: September, October, November

    Args:
        y_true: Ground truth values, shape (N, T, V)
        y_pred: Predicted values, shape (N, T, V)
        mask: Evaluation mask, shape (N, T, V), True=evaluate
        timestamps: Unix timestamps for each observation, shape (N, T)
                   Must be in seconds since epoch

    Returns:
        Dictionary mapping season name to metrics dict.
        Keys: "winter", "spring", "summer", "fall"

    Example:
        >>> import datetime
        >>> y_true = torch.randn(10, 100, 6)
        >>> y_pred = y_true + torch.randn(10, 100, 6) * 0.1
        >>> mask = torch.rand(10, 100, 6) < 0.3
        >>> # Create timestamps spanning a year
        >>> start = datetime.datetime(2023, 1, 1).timestamp()
        >>> timestamps = torch.arange(10 * 100).float() * 3600 + start
        >>> timestamps = timestamps.reshape(10, 100)
        >>> results = stratify_by_season(y_true, y_pred, mask, timestamps)
    """
    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, "
            f"mask {mask.shape}"
        )

    if timestamps.shape != y_true.shape[:2]:
        raise ValueError(
            f"timestamps shape {timestamps.shape} must match "
            f"(N, T) = {y_true.shape[:2]}"
        )

    # Convert Unix timestamps to datetime components
    # timestamps are in seconds since epoch
    # We need month (1-12) for season determination

    # Create datetime from timestamps using torch operations
    # This is a simplified approach that works for timestamps in seconds
    import datetime

    # Extract months from timestamps (1-12)
    # Using numpy/datetime for simplicity since PyTorch doesn't have datetime ops
    timestamps_np = timestamps.cpu().numpy()
    months = torch.zeros_like(timestamps, dtype=torch.long)

    for i in range(timestamps.shape[0]):
        for j in range(timestamps.shape[1]):
            # Convert float timestamp to int for datetime
            timestamp_int = int(timestamps_np[i, j])
            dt = datetime.datetime.fromtimestamp(timestamp_int)
            months[i, j] = dt.month

    # Define season mappings
    seasons = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "fall": [9, 10, 11],
    }

    results = {}

    for season_name, season_months in seasons.items():
        # Create season mask: True for timesteps in this season
        season_mask_time = torch.zeros_like(months, dtype=torch.bool)
        for month in season_months:
            season_mask_time |= months == month

        # Broadcast to (N, T, V) and combine with evaluation mask
        season_mask = season_mask_time.unsqueeze(-1).expand_as(mask) & mask

        if season_mask.any():
            metrics = compute_all_metrics(y_true, y_pred, season_mask)
            results[season_name] = metrics

    return results


def compute_stratified_metrics(
    y_true: Tensor,
    y_pred: Tensor,
    mask: Tensor,
    gap_lengths: Tensor | None = None,
    timestamps: Tensor | None = None,
    variable_names: list[str] | None = None,
    gap_bins: list[int] | None = None,
    extreme_percentiles: tuple[float, float] = (5.0, 95.0),
    stratify_gap_length: bool = True,
    stratify_season: bool = False,
    stratify_variable: bool = True,
    stratify_extremes: bool = True,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute all stratified metrics in one call.

    This is a convenience function that computes multiple stratifications
    and returns them in a nested dictionary structure.

    Args:
        y_true: Ground truth values, shape (N, T, V)
        y_pred: Predicted values, shape (N, T, V)
        mask: Evaluation mask, shape (N, T, V), True=evaluate
        gap_lengths: Gap lengths tensor, required if stratify_gap_length=True
        timestamps: Unix timestamps, required if stratify_season=True
        variable_names: Variable names for stratify_variable
        gap_bins: Custom gap length bins
        extreme_percentiles: Percentile thresholds for extremes
        stratify_gap_length: Compute gap length stratification
        stratify_season: Compute seasonal stratification
        stratify_variable: Compute per-variable stratification
        stratify_extremes: Compute extreme value stratification

    Returns:
        Nested dictionary with structure:
        {
            "gap_length": {"1-6h": {...}, "6-24h": {...}, ...},
            "variable": {"temperature": {...}, "pressure": {...}, ...},
            "extremes": {"extreme_low": {...}, "normal": {...}, ...},
            "season": {"winter": {...}, "spring": {...}, ...}
        }

    Raises:
        ValueError: If required inputs are missing for requested stratifications
    """
    results = {}

    if stratify_gap_length:
        if gap_lengths is None:
            raise ValueError("gap_lengths required for gap length stratification")
        results["gap_length"] = stratify_by_gap_length(
            y_true, y_pred, mask, gap_lengths, bins=gap_bins
        )

    if stratify_variable:
        results["variable"] = stratify_by_variable(
            y_true, y_pred, mask, variable_names
        )

    if stratify_extremes:
        results["extremes"] = stratify_by_extreme_values(
            y_true, y_pred, mask, percentiles=extreme_percentiles
        )

    if stratify_season:
        if timestamps is None:
            raise ValueError("timestamps required for seasonal stratification")
        results["season"] = stratify_by_season(y_true, y_pred, mask, timestamps)

    return results
