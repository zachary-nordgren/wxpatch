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
        raise NotImplementedError("MAR masking strategy not yet implemented")
    elif config.strategy == "mnar":
        raise NotImplementedError("MNAR masking strategy not yet implemented")
    elif config.strategy == "realistic":
        raise NotImplementedError("Realistic masking strategy not yet implemented")
    else:
        raise ValueError(f"Unknown masking strategy: {config.strategy}")
