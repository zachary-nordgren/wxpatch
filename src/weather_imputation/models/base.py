"""Base protocol for imputation models."""

from abc import ABC, abstractmethod
from typing import Any

import polars as pl


class BaseImputer(ABC):
    """Abstract base class for weather data imputation methods.

    All imputation models should inherit from this class and implement
    the required methods.
    """

    @abstractmethod
    def fit(self, df: pl.DataFrame, target_column: str) -> "BaseImputer":
        """Fit the imputer to training data.

        Args:
            df: Training DataFrame with weather data
            target_column: Name of the column to impute

        Returns:
            Self for method chaining
        """
        ...

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply imputation to fill missing values.

        Args:
            df: DataFrame with missing values to impute

        Returns:
            DataFrame with missing values filled
        """
        ...

    def fit_transform(self, df: pl.DataFrame, target_column: str) -> pl.DataFrame:
        """Fit and transform in one step.

        Args:
            df: DataFrame with weather data
            target_column: Name of the column to impute

        Returns:
            DataFrame with missing values filled
        """
        return self.fit(df, target_column).transform(df)

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the imputation method."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return False
