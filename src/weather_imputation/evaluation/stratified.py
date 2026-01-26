"""Stratified evaluation by gap length, season, and other factors."""

from dataclasses import dataclass
from enum import Enum

import polars as pl

from weather_imputation.evaluation.metrics import ImputationMetrics, compute_all_metrics


class Season(Enum):
    """Meteorological seasons for stratified analysis."""

    WINTER = "winter"  # Dec, Jan, Feb
    SPRING = "spring"  # Mar, Apr, May
    SUMMER = "summer"  # Jun, Jul, Aug
    FALL = "fall"  # Sep, Oct, Nov


@dataclass
class StratifiedResult:
    """Metrics stratified by a grouping factor."""

    group_name: str
    group_value: str
    n_samples: int
    metrics: ImputationMetrics

    def to_dict(self) -> dict[str, str | int | float | None]:
        return {
            "group_name": self.group_name,
            "group_value": self.group_value,
            "n_samples": self.n_samples,
            **self.metrics.to_dict(),
        }


def get_season(month: int) -> Season:
    """Get season for a given month.

    Args:
        month: Month number (1-12)

    Returns:
        Season enum value
    """
    if month in (12, 1, 2):
        return Season.WINTER
    elif month in (3, 4, 5):
        return Season.SPRING
    elif month in (6, 7, 8):
        return Season.SUMMER
    else:
        return Season.FALL


def stratify_by_gap_length(
    df: pl.DataFrame,
    actual_col: str,
    predicted_col: str,
    gap_length_col: str,
    bins: list[int] | None = None,
) -> list[StratifiedResult]:
    """Compute metrics stratified by gap length.

    Args:
        df: DataFrame with predictions
        actual_col: Column with true values
        predicted_col: Column with predicted values
        gap_length_col: Column with gap lengths (hours)
        bins: Gap length bins (e.g., [1, 6, 24, 72])

    Returns:
        List of StratifiedResult for each gap length bin
    """
    if bins is None:
        bins = [1, 6, 12, 24, 48, 72, 168]  # 1h, 6h, 12h, 1d, 2d, 3d, 1w

    results = []

    for i in range(len(bins)):
        if i == 0:
            lower, upper = 0, bins[0]
            label = f"<{bins[0]}h"
        else:
            lower, upper = bins[i - 1], bins[i]
            label = f"{lower}-{upper}h"

        # Filter to this gap length range
        mask = (pl.col(gap_length_col) > lower) & (pl.col(gap_length_col) <= upper)
        subset = df.filter(mask)

        if len(subset) == 0:
            continue

        metrics = compute_all_metrics(
            subset[actual_col],
            subset[predicted_col],
        )

        results.append(StratifiedResult(
            group_name="gap_length",
            group_value=label,
            n_samples=len(subset),
            metrics=metrics,
        ))

    # Add >max bin
    max_bin = bins[-1]
    mask = pl.col(gap_length_col) > max_bin
    subset = df.filter(mask)
    if len(subset) > 0:
        metrics = compute_all_metrics(subset[actual_col], subset[predicted_col])
        results.append(StratifiedResult(
            group_name="gap_length",
            group_value=f">{max_bin}h",
            n_samples=len(subset),
            metrics=metrics,
        ))

    return results


def stratify_by_season(
    df: pl.DataFrame,
    actual_col: str,
    predicted_col: str,
    date_col: str = "DATE",
) -> list[StratifiedResult]:
    """Compute metrics stratified by season.

    Args:
        df: DataFrame with predictions
        actual_col: Column with true values
        predicted_col: Column with predicted values
        date_col: Column with datetime values

    Returns:
        List of StratifiedResult for each season
    """
    results = []

    # Add month column
    df_with_month = df.with_columns(
        pl.col(date_col).dt.month().alias("_month")
    )

    for season in Season:
        if season == Season.WINTER:
            months = [12, 1, 2]
        elif season == Season.SPRING:
            months = [3, 4, 5]
        elif season == Season.SUMMER:
            months = [6, 7, 8]
        else:
            months = [9, 10, 11]

        subset = df_with_month.filter(pl.col("_month").is_in(months))

        if len(subset) == 0:
            continue

        metrics = compute_all_metrics(
            subset[actual_col],
            subset[predicted_col],
        )

        results.append(StratifiedResult(
            group_name="season",
            group_value=season.value,
            n_samples=len(subset),
            metrics=metrics,
        ))

    return results


def stratify_by_hour(
    df: pl.DataFrame,
    actual_col: str,
    predicted_col: str,
    date_col: str = "DATE",
) -> list[StratifiedResult]:
    """Compute metrics stratified by hour of day.

    Args:
        df: DataFrame with predictions
        actual_col: Column with true values
        predicted_col: Column with predicted values
        date_col: Column with datetime values

    Returns:
        List of StratifiedResult for each hour
    """
    results = []

    df_with_hour = df.with_columns(
        pl.col(date_col).dt.hour().alias("_hour")
    )

    for hour in range(24):
        subset = df_with_hour.filter(pl.col("_hour") == hour)

        if len(subset) == 0:
            continue

        metrics = compute_all_metrics(
            subset[actual_col],
            subset[predicted_col],
        )

        results.append(StratifiedResult(
            group_name="hour",
            group_value=f"{hour:02d}:00",
            n_samples=len(subset),
            metrics=metrics,
        ))

    return results


def stratify_by_value_range(
    df: pl.DataFrame,
    actual_col: str,
    predicted_col: str,
    bins: list[float] | None = None,
) -> list[StratifiedResult]:
    """Compute metrics stratified by actual value range.

    Useful for understanding if imputation quality varies with
    temperature extremes, for example.

    Args:
        df: DataFrame with predictions
        actual_col: Column with true values
        predicted_col: Column with predicted values
        bins: Value bins (e.g., [-20, 0, 10, 20, 30, 40] for temperature)

    Returns:
        List of StratifiedResult for each value range
    """
    if bins is None:
        # Default temperature bins (Celsius)
        bins = [-30, -10, 0, 10, 20, 30, 40]

    results = []

    for i in range(len(bins) - 1):
        lower, upper = bins[i], bins[i + 1]
        label = f"{lower} to {upper}"

        mask = (pl.col(actual_col) >= lower) & (pl.col(actual_col) < upper)
        subset = df.filter(mask)

        if len(subset) == 0:
            continue

        metrics = compute_all_metrics(
            subset[actual_col],
            subset[predicted_col],
        )

        results.append(StratifiedResult(
            group_name="value_range",
            group_value=label,
            n_samples=len(subset),
            metrics=metrics,
        ))

    return results
