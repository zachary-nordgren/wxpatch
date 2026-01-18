"""Spline interpolation imputation method."""

from typing import Any

import polars as pl

from weather_imputation.models.base import BaseImputer


class AkimaSplineImputer(BaseImputer):
    """Impute missing values using Akima spline interpolation.

    Akima splines are a type of cubic spline that produces smoother
    interpolations by using a weighted average of slopes. They are
    less prone to oscillation than standard cubic splines.

    Suitable for:
    - Smooth continuous variables (temperature, pressure)
    - Medium-length gaps

    Not suitable for:
    - Very long gaps
    - Highly discontinuous data
    """

    def __init__(self, max_gap_hours: int = 12):
        """Initialize the Akima spline imputer.

        Args:
            max_gap_hours: Maximum gap size to interpolate
        """
        self.max_gap_hours = max_gap_hours
        self._target_column: str | None = None
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "Akima Spline"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def get_params(self) -> dict[str, Any]:
        return {"max_gap_hours": self.max_gap_hours}

    def fit(self, df: pl.DataFrame, target_column: str) -> "AkimaSplineImputer":
        """Fit the imputer.

        Args:
            df: Training DataFrame
            target_column: Column to impute

        Returns:
            Self
        """
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in DataFrame")

        self._target_column = target_column
        self._is_fitted = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply Akima spline interpolation.

        Note: This is a placeholder implementation. Full Akima spline
        interpolation requires scipy or a custom implementation.

        Args:
            df: DataFrame with missing values

        Returns:
            DataFrame with interpolated values
        """
        if not self._is_fitted or self._target_column is None:
            raise ValueError("Imputer must be fitted before transform")

        # Placeholder: use linear interpolation for now
        # TODO: Implement actual Akima spline using scipy.interpolate.Akima1DInterpolator
        return df.with_columns(
            pl.col(self._target_column).interpolate().alias(self._target_column)
        )
