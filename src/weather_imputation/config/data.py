"""Data configuration classes for weather imputation.

This module defines Pydantic models for configuring data processing,
including station filtering, normalization, masking, and splitting strategies.
"""

from typing import Literal

from pydantic import Field, field_validator

from weather_imputation.config.base import BaseConfig


class StationFilterConfig(BaseConfig):
    """Configuration for filtering weather stations by quality thresholds.

    Filters stations based on data completeness, temporal coverage,
    and geographic location to ensure sufficient quality for imputation.

    Example:
        >>> filter_config = StationFilterConfig(
        ...     min_temperature_completeness=70.0,
        ...     min_years_available=5,
        ...     latitude_range=(24.0, 50.0)
        ... )
    """

    # Completeness thresholds (percentage of non-null observations)
    min_temperature_completeness: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Minimum temperature completeness percentage (0-100)",
    )
    min_dew_point_completeness: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Minimum dew point completeness percentage (0-100)",
    )
    min_sea_level_pressure_completeness: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Minimum sea level pressure completeness percentage (0-100)",
    )
    min_wind_speed_completeness: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Minimum wind speed completeness percentage (0-100)",
    )
    min_wind_direction_completeness: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Minimum wind direction completeness percentage (0-100)",
    )
    min_relative_humidity_completeness: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Minimum relative humidity completeness percentage (0-100)",
    )

    # Temporal coverage thresholds
    min_years_available: int = Field(
        default=3,
        ge=1,
        description="Minimum number of years with available data",
    )
    min_total_observations: int = Field(
        default=1000,
        ge=1,
        description="Minimum total number of observations",
    )

    # Geographic bounds (North America default)
    latitude_range: tuple[float, float] = Field(
        default=(24.0, 50.0),
        description="Valid latitude range (min, max) in decimal degrees",
    )
    longitude_range: tuple[float, float] = Field(
        default=(-130.0, -60.0),
        description="Valid longitude range (min, max) in decimal degrees",
    )

    # Gap pattern thresholds
    max_gap_duration_hours: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum allowed gap duration in hours (None = no limit)",
    )

    @field_validator("latitude_range", "longitude_range")
    @classmethod
    def validate_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Validate that range tuple has min < max.

        Args:
            v: Range tuple (min, max)

        Returns:
            Validated range tuple

        Raises:
            ValueError: If min >= max
        """
        if v[0] >= v[1]:
            raise ValueError(f"Range minimum ({v[0]}) must be less than maximum ({v[1]})")
        return v


class NormalizationConfig(BaseConfig):
    """Configuration for variable normalization.

    Specifies the normalization method and whether to normalize per-station
    or globally across all stations.

    Example:
        >>> norm_config = NormalizationConfig(
        ...     method="zscore",
        ...     per_station=True
        ... )
    """

    method: Literal["zscore", "minmax", "none"] = Field(
        default="zscore",
        description="Normalization method: zscore (mean=0, std=1), minmax (0-1 range), or none",
    )
    per_station: bool = Field(
        default=True,
        description="Whether to normalize per-station (True) or globally (False)",
    )
    clip_outliers: bool = Field(
        default=False,
        description="Whether to clip outliers beyond 5 standard deviations (zscore only)",
    )


class MaskingConfig(BaseConfig):
    """Configuration for synthetic gap generation strategies.

    Defines how to create artificial missing data patterns for controlled
    evaluation of imputation methods.

    Example:
        >>> mask_config = MaskingConfig(
        ...     strategy="mcar",
        ...     missing_ratio=0.2
        ... )
    """

    strategy: Literal["mcar", "mar", "mnar", "realistic"] = Field(
        default="realistic",
        description=(
            "Gap generation strategy: "
            "mcar (missing completely at random), "
            "mar (missing at random, depends on observed values), "
            "mnar (missing not at random, depends on missing values), "
            "realistic (based on observed gap patterns)"
        ),
    )
    missing_ratio: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Target proportion of missing values (0.0-1.0)",
    )
    min_gap_length: int = Field(
        default=1,
        ge=1,
        description="Minimum gap length in timesteps",
    )
    max_gap_length: int = Field(
        default=168,
        ge=1,
        description="Maximum gap length in timesteps (168 hours = 1 week)",
    )

    @field_validator("max_gap_length")
    @classmethod
    def validate_gap_length(cls, v: int, info) -> int:
        """Validate that max_gap_length >= min_gap_length.

        Args:
            v: Maximum gap length
            info: Validation info containing other field values

        Returns:
            Validated max gap length

        Raises:
            ValueError: If max < min
        """
        min_gap = info.data.get("min_gap_length", 1)
        if v < min_gap:
            raise ValueError(
                f"max_gap_length ({v}) must be >= min_gap_length ({min_gap})"
            )
        return v


class SplitConfig(BaseConfig):
    """Configuration for train/validation/test splitting.

    Defines the strategy and ratios for splitting data into training,
    validation, and test sets.

    Example:
        >>> split_config = SplitConfig(
        ...     strategy="simulated",
        ...     train_ratio=0.7,
        ...     val_ratio=0.15,
        ...     test_ratio=0.15
        ... )
    """

    strategy: Literal["spatial", "temporal", "hybrid", "simulated"] = Field(
        default="simulated",
        description=(
            "Split strategy: "
            "spatial (split by stations), "
            "temporal (split by time), "
            "hybrid (combination), "
            "simulated (Strategy D: simulate real missing patterns within signals)"
        ),
    )
    train_ratio: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Proportion of data for training (0.0-1.0)",
    )
    val_ratio: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Proportion of data for validation (0.0-1.0)",
    )
    test_ratio: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Proportion of data for testing (0.0-1.0)",
    )

    @field_validator("test_ratio")
    @classmethod
    def validate_ratio_sum(cls, v: float, info) -> float:
        """Validate that train + val + test ratios sum to 1.0.

        Args:
            v: Test ratio
            info: Validation info containing other field values

        Returns:
            Validated test ratio

        Raises:
            ValueError: If ratios don't sum to 1.0 (within tolerance)
        """
        train = info.data.get("train_ratio", 0.7)
        val = info.data.get("val_ratio", 0.15)
        total = train + val + v

        if not 0.99 <= total <= 1.01:  # Allow small floating point tolerance
            raise ValueError(
                f"train_ratio ({train}) + val_ratio ({val}) + test_ratio ({v}) "
                f"must sum to 1.0 (got {total})"
            )
        return v


class DataConfig(BaseConfig):
    """Main data configuration for weather imputation.

    Aggregates all data processing configurations including variable selection,
    station filtering, normalization, masking, and splitting.

    Example:
        >>> data_config = DataConfig(
        ...     variables=["temperature", "dew_point_temperature"],
        ...     station_filter=StationFilterConfig(min_temperature_completeness=70.0),
        ...     window_size=168
        ... )
    """

    # Tier 1 variables (6 core continuous variables)
    variables: list[str] = Field(
        default=[
            "temperature",
            "dew_point_temperature",
            "sea_level_pressure",
            "wind_speed",
            "wind_direction",
            "relative_humidity",
        ],
        min_length=1,
        description="List of weather variables to include (Tier 1 variables)",
    )

    # Sub-configurations
    station_filter: StationFilterConfig = Field(
        default_factory=StationFilterConfig,
        description="Station quality filtering configuration",
    )
    normalization: NormalizationConfig = Field(
        default_factory=NormalizationConfig,
        description="Variable normalization configuration",
    )
    masking: MaskingConfig = Field(
        default_factory=MaskingConfig,
        description="Synthetic gap generation configuration",
    )
    split: SplitConfig = Field(
        default_factory=SplitConfig,
        description="Train/validation/test split configuration",
    )

    # Time series windowing
    window_size: int = Field(
        default=168,
        ge=1,
        description="Time series window size in hours (168 = 1 week)",
    )
    stride: int = Field(
        default=24,
        ge=1,
        description="Stride between consecutive windows in hours",
    )

    # Quality control
    report_types: list[str] = Field(
        default=["AUTO", "FM-15"],
        min_length=1,
        description="Allowed GHCNh report types for filtering",
    )

    @field_validator("variables")
    @classmethod
    def validate_variables(cls, v: list[str]) -> list[str]:
        """Validate that variables are from the Tier 1 set.

        Args:
            v: List of variable names

        Returns:
            Validated variable list

        Raises:
            ValueError: If any variable is not in Tier 1
        """
        tier1_variables = {
            "temperature",
            "dew_point_temperature",
            "sea_level_pressure",
            "wind_speed",
            "wind_direction",
            "relative_humidity",
        }

        invalid_vars = [var for var in v if var not in tier1_variables]
        if invalid_vars:
            raise ValueError(
                f"Invalid variables {invalid_vars}. Must be from Tier 1: {tier1_variables}"
            )

        return v
