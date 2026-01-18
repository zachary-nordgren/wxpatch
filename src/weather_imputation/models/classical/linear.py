"""Linear interpolation imputation method."""

from typing import Any

import polars as pl

from weather_imputation.models.base import BaseImputer


class LinearInterpolationImputer(BaseImputer):
    """Impute missing values using linear interpolation.

    This method fills missing values by drawing a straight line between
    the nearest non-null values on either side of the gap.

    Suitable for:
    - Short gaps in continuous variables (temperature, pressure)
    - Data with gradual changes

    Not suitable for:
    - Long gaps (>24 hours)
    - Highly variable data (wind direction, precipitation)
    """

    def __init__(self, max_gap_hours: int = 6):
        """Initialize the linear interpolation imputer.

        Args:
            max_gap_hours: Maximum gap size to interpolate (larger gaps remain null)
        """
        self.max_gap_hours = max_gap_hours
        self._target_column: str | None = None
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "Linear Interpolation"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def get_params(self) -> dict[str, Any]:
        return {"max_gap_hours": self.max_gap_hours}

    def fit(self, df: pl.DataFrame, target_column: str) -> "LinearInterpolationImputer":
        """Fit the imputer (stores target column name).

        Linear interpolation is a non-parametric method, so fitting
        just validates the input.

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
        """Apply linear interpolation to fill missing values.

        Args:
            df: DataFrame with missing values

        Returns:
            DataFrame with interpolated values
        """
        if not self._is_fitted or self._target_column is None:
            raise ValueError("Imputer must be fitted before transform")

        col = self._target_column

        # Use Polars' built-in interpolation
        # Note: This does basic linear interpolation without gap limits
        result = df.with_columns(
            pl.col(col).interpolate().alias(col)
        )

        # TODO: Implement max_gap_hours limit
        # This would require identifying gap lengths and
        # resetting interpolated values for large gaps

        return result
