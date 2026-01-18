"""Evaluation metrics for imputation quality."""

import math
from dataclasses import dataclass

import polars as pl


@dataclass
class ImputationMetrics:
    """Container for imputation evaluation metrics."""

    rmse: float  # Root Mean Squared Error
    mae: float  # Mean Absolute Error
    mape: float | None  # Mean Absolute Percentage Error (None if zeros present)
    r2: float  # R-squared (coefficient of determination)
    bias: float  # Mean error (positive = overestimate)
    coverage: float  # Fraction of values successfully imputed

    def to_dict(self) -> dict[str, float | None]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r2": self.r2,
            "bias": self.bias,
            "coverage": self.coverage,
        }


def rmse(actual: pl.Series, predicted: pl.Series) -> float:
    """Calculate Root Mean Squared Error.

    Args:
        actual: True values
        predicted: Predicted/imputed values

    Returns:
        RMSE value
    """
    # Filter to non-null pairs
    mask = actual.is_not_null() & predicted.is_not_null()
    a = actual.filter(mask)
    p = predicted.filter(mask)

    if len(a) == 0:
        return float("nan")

    squared_errors = (a - p) ** 2
    return math.sqrt(squared_errors.mean())


def mae(actual: pl.Series, predicted: pl.Series) -> float:
    """Calculate Mean Absolute Error.

    Args:
        actual: True values
        predicted: Predicted/imputed values

    Returns:
        MAE value
    """
    mask = actual.is_not_null() & predicted.is_not_null()
    a = actual.filter(mask)
    p = predicted.filter(mask)

    if len(a) == 0:
        return float("nan")

    return (a - p).abs().mean()


def mape(actual: pl.Series, predicted: pl.Series) -> float | None:
    """Calculate Mean Absolute Percentage Error.

    Args:
        actual: True values
        predicted: Predicted/imputed values

    Returns:
        MAPE value as percentage, or None if actual contains zeros
    """
    mask = actual.is_not_null() & predicted.is_not_null() & (actual != 0)
    a = actual.filter(mask)
    p = predicted.filter(mask)

    if len(a) == 0:
        return None

    percentage_errors = ((a - p).abs() / a.abs()) * 100
    return percentage_errors.mean()


def r_squared(actual: pl.Series, predicted: pl.Series) -> float:
    """Calculate R-squared (coefficient of determination).

    Args:
        actual: True values
        predicted: Predicted/imputed values

    Returns:
        R² value (1.0 = perfect, 0.0 = baseline, negative = worse than mean)
    """
    mask = actual.is_not_null() & predicted.is_not_null()
    a = actual.filter(mask)
    p = predicted.filter(mask)

    if len(a) == 0:
        return float("nan")

    # Total sum of squares
    mean_actual = a.mean()
    ss_tot = ((a - mean_actual) ** 2).sum()

    # Residual sum of squares
    ss_res = ((a - p) ** 2).sum()

    if ss_tot == 0:
        return float("nan")

    return 1 - (ss_res / ss_tot)


def bias(actual: pl.Series, predicted: pl.Series) -> float:
    """Calculate mean bias (systematic error).

    Positive bias means predictions are systematically too high.

    Args:
        actual: True values
        predicted: Predicted/imputed values

    Returns:
        Mean bias
    """
    mask = actual.is_not_null() & predicted.is_not_null()
    a = actual.filter(mask)
    p = predicted.filter(mask)

    if len(a) == 0:
        return float("nan")

    return (p - a).mean()


def coverage(original: pl.Series, imputed: pl.Series) -> float:
    """Calculate imputation coverage.

    Coverage is the fraction of originally missing values that were imputed.

    Args:
        original: Original series with missing values
        imputed: Series after imputation

    Returns:
        Coverage as fraction (0-1)
    """
    originally_missing = original.null_count()
    if originally_missing == 0:
        return 1.0

    still_missing = imputed.null_count()
    imputed_count = originally_missing - still_missing

    return imputed_count / originally_missing


def crps(actual: pl.Series, predicted_samples: list[pl.Series]) -> float:
    """Calculate Continuous Ranked Probability Score.

    CRPS measures the quality of probabilistic predictions by comparing
    the predicted distribution to the actual value.

    Args:
        actual: True values
        predicted_samples: List of sample predictions (ensemble)

    Returns:
        Mean CRPS value (lower is better)
    """
    # TODO: Implement full CRPS calculation
    # For now, return NaN as placeholder
    return float("nan")


def compute_all_metrics(
    actual: pl.Series,
    predicted: pl.Series,
    original: pl.Series | None = None,
) -> ImputationMetrics:
    """Compute all standard imputation metrics.

    Args:
        actual: True values (before gaps were introduced)
        predicted: Predicted/imputed values
        original: Original series with gaps (for coverage calculation)

    Returns:
        ImputationMetrics dataclass
    """
    return ImputationMetrics(
        rmse=rmse(actual, predicted),
        mae=mae(actual, predicted),
        mape=mape(actual, predicted),
        r2=r_squared(actual, predicted),
        bias=bias(actual, predicted),
        coverage=coverage(original, predicted) if original is not None else 1.0,
    )
