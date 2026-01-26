"""Circular statistics utilities for wind direction.

Wind direction is a circular variable (0° = 360°) requiring special handling.
This module provides encoding, decoding, and statistical functions for circular data.

References:
    - Mardia, K. V., & Jupp, P. E. (2000). Directional Statistics. Wiley.
    - Fisher, N. I. (1993). Statistical Analysis of Circular Data. Cambridge University Press.
"""

import math
from typing import Literal

import torch


def degrees_to_radians(degrees: torch.Tensor) -> torch.Tensor:
    """Convert degrees to radians.

    Args:
        degrees: Wind direction in degrees [0, 360).

    Returns:
        Wind direction in radians [0, 2π).
    """
    return degrees * (math.pi / 180.0)


def radians_to_degrees(radians: torch.Tensor) -> torch.Tensor:
    """Convert radians to degrees.

    Args:
        radians: Wind direction in radians.

    Returns:
        Wind direction in degrees [0, 360).
    """
    degrees = radians * (180.0 / math.pi)
    # Normalize to [0, 360)
    return torch.remainder(degrees, 360.0)


def encode_wind_direction(
    degrees: torch.Tensor, method: Literal["sincos"] = "sincos"
) -> torch.Tensor:
    """Encode wind direction as Cartesian components.

    Converts circular wind direction (0-360°) to Cartesian coordinates.
    This encoding is continuous at the 0°/360° boundary and suitable for
    neural networks.

    Args:
        degrees: Wind direction in degrees [0, 360). Shape: (...,)
        method: Encoding method. Currently only "sincos" is supported.
            - "sincos": Encode as (sin θ, cos θ) pair.

    Returns:
        Encoded wind direction. Shape: (..., 2)
            - First channel: sin(θ)
            - Second channel: cos(θ)

    Examples:
        >>> degrees = torch.tensor([0.0, 90.0, 180.0, 270.0])
        >>> encoded = encode_wind_direction(degrees)
        >>> encoded.shape
        torch.Size([4, 2])
        >>> # North (0°): (sin=0, cos=1), East (90°): (sin=1, cos=0)
    """
    if method != "sincos":
        msg = f"Unknown encoding method: {method}. Only 'sincos' is supported."
        raise ValueError(msg)

    radians = degrees_to_radians(degrees)
    sin_component = torch.sin(radians)
    cos_component = torch.cos(radians)

    # Stack along last dimension: (..., 2)
    return torch.stack([sin_component, cos_component], dim=-1)


def decode_wind_direction(encoded: torch.Tensor) -> torch.Tensor:
    """Decode wind direction from Cartesian components.

    Converts (sin θ, cos θ) encoding back to degrees [0, 360).

    Args:
        encoded: Encoded wind direction. Shape: (..., 2)
            - First channel: sin(θ)
            - Second channel: cos(θ)

    Returns:
        Wind direction in degrees [0, 360). Shape: (...,)

    Examples:
        >>> encoded = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        >>> degrees = decode_wind_direction(encoded)
        >>> degrees
        tensor([0., 90.])
    """
    if encoded.shape[-1] != 2:
        msg = f"Expected last dimension to be 2 (sin, cos), got {encoded.shape[-1]}"
        raise ValueError(msg)

    sin_component = encoded[..., 0]
    cos_component = encoded[..., 1]

    # atan2 returns [-π, π], convert to [0, 2π) then to degrees
    radians = torch.atan2(sin_component, cos_component)
    return radians_to_degrees(radians)


def circular_mean(degrees: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute circular mean of wind directions.

    The circular mean is computed by averaging the unit vectors and
    converting back to degrees.

    Args:
        degrees: Wind directions in degrees. Shape: (N,) or (..., N)
        mask: Optional boolean mask indicating valid values (True = valid).
            Shape must broadcast with degrees. If None, all values are valid.

    Returns:
        Circular mean in degrees [0, 360). Scalar if input is 1D,
        otherwise shape matches input dimensions except the last.

    Examples:
        >>> # Mean of north (0°) and east (90°) is northeast (45°)
        >>> degrees = torch.tensor([0.0, 90.0])
        >>> circular_mean(degrees)
        tensor(45.)
        >>> # Mean of north (0°) and north (360°) is north (0°)
        >>> degrees = torch.tensor([0.0, 360.0])
        >>> circular_mean(degrees)
        tensor(0.)
    """
    radians = degrees_to_radians(degrees)
    sin_sum = torch.sin(radians)
    cos_sum = torch.cos(radians)

    if mask is not None:
        # Apply mask (set invalid values to 0 contribution)
        sin_sum = torch.where(mask, sin_sum, torch.zeros_like(sin_sum))
        cos_sum = torch.where(mask, cos_sum, torch.zeros_like(cos_sum))
        # Sum over last dimension
        sin_sum = sin_sum.sum(dim=-1)
        cos_sum = cos_sum.sum(dim=-1)
    else:
        # Sum over last dimension
        sin_sum = sin_sum.sum(dim=-1)
        cos_sum = cos_sum.sum(dim=-1)

    # Compute mean angle
    mean_radians = torch.atan2(sin_sum, cos_sum)
    return radians_to_degrees(mean_radians)


def circular_std(degrees: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute circular standard deviation of wind directions.

    Circular standard deviation is computed from the mean resultant length R:
        σ = sqrt(-2 * ln(R))

    This is a measure of angular dispersion. Values near 0 indicate
    concentration, values near π indicate uniform distribution.

    Args:
        degrees: Wind directions in degrees. Shape: (N,) or (..., N)
        mask: Optional boolean mask indicating valid values (True = valid).
            Shape must broadcast with degrees. If None, all values are valid.

    Returns:
        Circular standard deviation in radians. Scalar if input is 1D,
        otherwise shape matches input dimensions except the last.

    Examples:
        >>> # Concentrated directions have low std
        >>> degrees = torch.tensor([0.0, 1.0, 359.0])
        >>> circular_std(degrees)
        tensor(0.0582)
        >>> # Dispersed directions have high std
        >>> degrees = torch.tensor([0.0, 90.0, 180.0, 270.0])
        >>> circular_std(degrees)
        tensor(1.2533)
    """
    radians = degrees_to_radians(degrees)
    sin_vals = torch.sin(radians)
    cos_vals = torch.cos(radians)

    if mask is not None:
        # Apply mask
        sin_vals = torch.where(mask, sin_vals, torch.zeros_like(sin_vals))
        cos_vals = torch.where(mask, cos_vals, torch.zeros_like(cos_vals))
        n_valid = mask.sum(dim=-1).float()
        # Compute mean resultant vector
        sin_mean = sin_vals.sum(dim=-1) / n_valid
        cos_mean = cos_vals.sum(dim=-1) / n_valid
    else:
        # Compute mean resultant vector
        sin_mean = sin_vals.mean(dim=-1)
        cos_mean = cos_vals.mean(dim=-1)

    # Mean resultant length (R)
    r = torch.sqrt(sin_mean**2 + cos_mean**2)

    # Clamp R to [0, 1] to avoid numerical issues with log
    r = torch.clamp(r, min=1e-7, max=1.0)

    # Circular standard deviation
    return torch.sqrt(-2.0 * torch.log(r))


def angular_difference(degrees1: torch.Tensor, degrees2: torch.Tensor) -> torch.Tensor:
    """Compute shortest angular difference between wind directions.

    Returns the signed difference in the range [-180, 180] degrees.
    Positive values indicate degrees2 is clockwise from degrees1.

    Args:
        degrees1: First wind direction in degrees. Shape: (...,)
        degrees2: Second wind direction in degrees. Shape: (...,)

    Returns:
        Angular difference in degrees [-180, 180]. Same shape as inputs.

    Examples:
        >>> angular_difference(torch.tensor(10.0), torch.tensor(350.0))
        tensor(-20.)
        >>> angular_difference(torch.tensor(350.0), torch.tensor(10.0))
        tensor(20.)
        >>> angular_difference(torch.tensor(0.0), torch.tensor(180.0))
        tensor(180.)
    """
    diff = degrees2 - degrees1
    # Normalize to [-180, 180]
    diff = torch.remainder(diff + 180.0, 360.0) - 180.0
    return diff
