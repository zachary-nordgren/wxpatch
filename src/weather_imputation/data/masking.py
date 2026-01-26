"""Masking strategies for generating synthetic gaps in time series data.

This module provides functions to create artificial missing data patterns for
controlled evaluation of imputation methods. Supports four strategies:

1. MCAR (Missing Completely At Random): Random uniform sampling
2. MAR (Missing At Random): Missingness depends on observed values
3. MNAR (Missing Not At Random): Missingness depends on unobserved values
4. Realistic: Based on observed gap patterns from real data

All masking functions work with PyTorch tensors and return boolean masks where
True indicates observed values and False indicates missing values.
"""

import logging

import numpy as np
import torch

from weather_imputation.config.data import MaskingConfig

logger = logging.getLogger(__name__)


def apply_mcar_mask(
    data: torch.Tensor,
    missing_ratio: float = 0.2,
    min_gap_length: int = 1,
    max_gap_length: int = 168,
    seed: int | None = None,
) -> torch.Tensor:
    """Apply Missing Completely At Random (MCAR) masking strategy.

    Generates random gaps with uniform probability across all timesteps and variables.
    The total missing ratio will be approximately equal to the specified missing_ratio.

    Args:
        data: Input tensor of shape (N, T, V) where N=samples, T=timesteps, V=variables
        missing_ratio: Target proportion of missing values (0.0-1.0)
        min_gap_length: Minimum gap length in timesteps
        max_gap_length: Maximum gap length in timesteps
        seed: Random seed for reproducibility (None = non-deterministic)

    Returns:
        Boolean mask tensor of shape (N, T, V) where True=observed, False=missing

    Raises:
        ValueError: If data is not 3D, missing_ratio not in [0, 1], or gap lengths invalid

    Example:
        >>> data = torch.randn(32, 168, 6)  # 32 samples, 168 hours, 6 variables
        >>> mask = apply_mcar_mask(data, missing_ratio=0.2, seed=42)
        >>> masked_data = data * mask  # Zero out missing values
        >>> mask.float().mean().item()  # Should be ~0.8 (1 - missing_ratio)
        0.802...
    """
    # Input validation
    if data.ndim != 3:
        raise ValueError(f"Expected 3D tensor (N, T, V), got shape {data.shape}")
    if not 0.0 <= missing_ratio <= 1.0:
        raise ValueError(f"missing_ratio must be in [0, 1], got {missing_ratio}")
    if min_gap_length < 1:
        raise ValueError(f"min_gap_length must be >= 1, got {min_gap_length}")
    if max_gap_length < min_gap_length:
        raise ValueError(
            f"max_gap_length ({max_gap_length}) must be >= min_gap_length ({min_gap_length})"
        )

    N, T, V = data.shape

    # Set random seed if provided
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    # Initialize mask as all observed (True)
    mask = torch.ones_like(data, dtype=torch.bool)

    # Calculate target number of missing values (timesteps × variables per sample)
    target_missing = int(missing_ratio * T * V)

    # Generate gaps for each sample independently
    for sample_idx in range(N):
        current_missing = 0
        max_iterations = target_missing * 20  # Safety limit to prevent infinite loops
        iterations = 0

        # Keep generating gaps until we reach target missing ratio
        while current_missing < target_missing and iterations < max_iterations:
            # Randomly select variable and start position
            var_idx = rng.randint(0, V)

            # To avoid overlap issues, try to find an observed position
            # For high missing ratios, we may need more attempts
            max_attempts = 50 if missing_ratio > 0.5 else 10
            found_observed = False
            for _ in range(max_attempts):
                start_idx = rng.randint(0, T)
                if mask[sample_idx, start_idx, var_idx]:
                    found_observed = True
                    break

            if not found_observed:
                # If we couldn't find observed position, we're likely near target
                break

            # Randomly select gap length
            gap_length = rng.randint(min_gap_length, max_gap_length + 1)

            # Limit gap length to not overshoot target by more than 50%
            remaining = target_missing - current_missing
            if gap_length > remaining * 1.5:
                gap_length = max(min_gap_length, int(remaining))

            # Calculate end position (clip to sequence length)
            end_idx = min(start_idx + gap_length, T)

            # Count how many NEW missing values this gap would create
            # (don't double-count already missing values)
            new_missing = mask[sample_idx, start_idx:end_idx, var_idx].sum().item()

            # Apply gap (set mask to False for missing values)
            mask[sample_idx, start_idx:end_idx, var_idx] = False

            current_missing += new_missing
            iterations += 1

        if iterations >= max_iterations:
            logger.debug(
                f"Sample {sample_idx}: Hit max iterations ({max_iterations}) "
                f"with {current_missing}/{target_missing} missing values"
            )

    # Log actual missing ratio
    actual_missing_ratio = (~mask).float().mean().item()
    logger.info(
        f"Applied MCAR mask: target={missing_ratio:.3f}, actual={actual_missing_ratio:.3f}"
    )

    return mask


def apply_mar_mask(
    data: torch.Tensor,
    missing_ratio: float = 0.2,
    min_gap_length: int = 1,
    max_gap_length: int = 168,
    condition_variable: int = 0,
    extreme_percentile: float = 0.15,
    seed: int | None = None,
) -> torch.Tensor:
    """Apply Missing At Random (MAR) masking strategy.

    Generates gaps where missingness probability depends on observed values.
    Specifically, missingness is more likely when the condition_variable has
    extreme values (below the lower extreme_percentile or above the upper
    extreme_percentile). This simulates realistic scenarios where sensors
    are more likely to fail during extreme weather conditions.

    Args:
        data: Input tensor of shape (N, T, V) where N=samples, T=timesteps, V=variables
        missing_ratio: Target proportion of missing values (0.0-1.0)
        min_gap_length: Minimum gap length in timesteps
        max_gap_length: Maximum gap length in timesteps
        condition_variable: Index of variable to condition missingness on (default: 0 = temperature)
        extreme_percentile: Percentile threshold for extreme values (default: 0.15 = bottom/top 15%)
        seed: Random seed for reproducibility (None = non-deterministic)

    Returns:
        Boolean mask tensor of shape (N, T, V) where True=observed, False=missing

    Raises:
        ValueError: If data is not 3D, missing_ratio not in [0, 1], gap lengths invalid,
                   or condition_variable out of range

    Example:
        >>> data = torch.randn(32, 168, 6)  # 32 samples, 168 hours, 6 variables
        >>> # Missingness more likely when variable 0 (temperature) is extreme
        >>> mask = apply_mar_mask(data, missing_ratio=0.2, condition_variable=0, seed=42)
        >>> # Check that extreme values have higher missingness
        >>> low_thresh = data[..., 0].quantile(0.15)
        >>> high_thresh = data[..., 0].quantile(0.85)
        >>> is_extreme = (data[..., 0] < low_thresh) | (data[..., 0] > high_thresh)
        >>> missing_at_extreme = (~mask[..., 0])[is_extreme].float().mean()
        >>> missing_at_normal = (~mask[..., 0])[~is_extreme].float().mean()
        >>> assert missing_at_extreme > missing_at_normal  # More missing at extremes
    """
    # Input validation
    if data.ndim != 3:
        raise ValueError(f"Expected 3D tensor (N, T, V), got shape {data.shape}")
    if not 0.0 <= missing_ratio <= 1.0:
        raise ValueError(f"missing_ratio must be in [0, 1], got {missing_ratio}")
    if min_gap_length < 1:
        raise ValueError(f"min_gap_length must be >= 1, got {min_gap_length}")
    if max_gap_length < min_gap_length:
        raise ValueError(
            f"max_gap_length ({max_gap_length}) must be >= min_gap_length ({min_gap_length})"
        )
    if not 0.0 <= extreme_percentile <= 0.5:
        raise ValueError(
            f"extreme_percentile must be in [0, 0.5], got {extreme_percentile}"
        )

    N, T, V = data.shape

    if condition_variable < 0 or condition_variable >= V:
        raise ValueError(
            f"condition_variable must be in [0, {V-1}], got {condition_variable}"
        )

    # Set random seed if provided
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    # Initialize mask as all observed (True)
    mask = torch.ones_like(data, dtype=torch.bool)

    # Calculate target number of missing values (timesteps × variables per sample)
    target_missing = int(missing_ratio * T * V)

    # Compute extreme value thresholds for the condition variable (across all samples)
    condition_values = data[..., condition_variable].flatten()
    lower_threshold = condition_values.quantile(extreme_percentile).item()
    upper_threshold = condition_values.quantile(1.0 - extreme_percentile).item()

    # Generate gaps for each sample independently
    for sample_idx in range(N):
        # Identify extreme timesteps for this sample (where condition variable is extreme)
        condition_var_values = data[sample_idx, :, condition_variable]
        is_extreme = (condition_var_values < lower_threshold) | (
            condition_var_values > upper_threshold
        )
        extreme_timesteps = torch.where(is_extreme)[0].numpy()
        normal_timesteps = torch.where(~is_extreme)[0].numpy()

        # If no extreme timesteps, fall back to uniform sampling
        if len(extreme_timesteps) == 0:
            logger.warning(
                f"Sample {sample_idx}: No extreme timesteps found, falling back to uniform sampling"
            )
            extreme_timesteps = np.arange(T)
            extreme_probability = 0.5
        else:
            # Bias probability towards extreme timesteps (3x more likely)
            extreme_probability = 0.75

        current_missing = 0
        max_iterations = target_missing * 20
        iterations = 0

        # Keep generating gaps until we reach target missing ratio
        while current_missing < target_missing and iterations < max_iterations:
            # Randomly select variable
            var_idx = rng.randint(0, V)

            # Select timestep with bias towards extreme conditions
            if rng.rand() < extreme_probability and len(extreme_timesteps) > 0:
                # Sample from extreme timesteps
                start_idx = int(rng.choice(extreme_timesteps))
            elif len(normal_timesteps) > 0:
                # Sample from normal timesteps
                start_idx = int(rng.choice(normal_timesteps))
            else:
                # Fallback to uniform if no normal timesteps
                start_idx = rng.randint(0, T)

            # Check if this position is already missing (to avoid excessive overlap)
            max_attempts = 50 if missing_ratio > 0.5 else 10
            found_observed = False
            for _ in range(max_attempts):
                if mask[sample_idx, start_idx, var_idx]:
                    found_observed = True
                    break
                # Try another position
                if rng.rand() < extreme_probability and len(extreme_timesteps) > 0:
                    start_idx = int(rng.choice(extreme_timesteps))
                elif len(normal_timesteps) > 0:
                    start_idx = int(rng.choice(normal_timesteps))
                else:
                    start_idx = rng.randint(0, T)

            if not found_observed:
                # If we couldn't find observed position, we're likely near target
                break

            # Randomly select gap length
            gap_length = rng.randint(min_gap_length, max_gap_length + 1)

            # Limit gap length to not overshoot target by more than 50%
            remaining = target_missing - current_missing
            if gap_length > remaining * 1.5:
                gap_length = max(min_gap_length, int(remaining))

            # Calculate end position (clip to sequence length)
            end_idx = min(start_idx + gap_length, T)

            # Count how many NEW missing values this gap would create
            new_missing = mask[sample_idx, start_idx:end_idx, var_idx].sum().item()

            # Apply gap (set mask to False for missing values)
            mask[sample_idx, start_idx:end_idx, var_idx] = False

            current_missing += new_missing
            iterations += 1

        if iterations >= max_iterations:
            logger.debug(
                f"Sample {sample_idx}: Hit max iterations ({max_iterations}) "
                f"with {current_missing}/{target_missing} missing values"
            )

    # Log actual missing ratio
    actual_missing_ratio = (~mask).float().mean().item()
    logger.info(
        f"Applied MAR mask: target={missing_ratio:.3f}, actual={actual_missing_ratio:.3f}, "
        f"condition_var={condition_variable}, extreme_pct={extreme_percentile:.2f}"
    )

    return mask


def apply_mask(
    data: torch.Tensor,
    config: MaskingConfig,
    seed: int | None = None,
) -> torch.Tensor:
    """Apply masking strategy based on configuration.

    Dispatches to the appropriate masking function based on config.strategy.

    Args:
        data: Input tensor of shape (N, T, V) where N=samples, T=timesteps, V=variables
        config: Masking configuration
        seed: Random seed for reproducibility (None = non-deterministic)

    Returns:
        Boolean mask tensor of shape (N, T, V) where True=observed, False=missing

    Raises:
        ValueError: If strategy is not supported or inputs are invalid

    Example:
        >>> from weather_imputation.config.data import MaskingConfig
        >>> config = MaskingConfig(strategy="mcar", missing_ratio=0.2)
        >>> data = torch.randn(32, 168, 6)
        >>> mask = apply_mask(data, config, seed=42)
    """
    if config.strategy == "mcar":
        return apply_mcar_mask(
            data,
            missing_ratio=config.missing_ratio,
            min_gap_length=config.min_gap_length,
            max_gap_length=config.max_gap_length,
            seed=seed,
        )
    elif config.strategy == "mar":
        return apply_mar_mask(
            data,
            missing_ratio=config.missing_ratio,
            min_gap_length=config.min_gap_length,
            max_gap_length=config.max_gap_length,
            seed=seed,
        )
    elif config.strategy == "mnar":
        raise NotImplementedError("MNAR masking strategy not yet implemented")
    elif config.strategy == "realistic":
        raise NotImplementedError("Realistic masking strategy not yet implemented")
    else:
        raise ValueError(f"Unknown masking strategy: {config.strategy}")
