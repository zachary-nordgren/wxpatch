"""Evaluation configuration classes."""

from typing import Literal

from pydantic import Field, field_validator

from weather_imputation.config.base import BaseConfig


class MetricConfig(BaseConfig):
    """Configuration for point and probabilistic metrics.

    Implements FR-011 (point metrics) and FR-012 (probabilistic metrics) from SPEC.md.
    """

    # Point metrics (RMSE, MAE, Bias, R²)
    compute_rmse: bool = Field(
        default=True,
        description="Compute Root Mean Squared Error on masked positions"
    )
    compute_mae: bool = Field(
        default=True,
        description="Compute Mean Absolute Error on masked positions"
    )
    compute_bias: bool = Field(
        default=True,
        description="Compute bias (mean error) on masked positions"
    )
    compute_r2: bool = Field(
        default=True,
        description="Compute R² (coefficient of determination) on masked positions"
    )

    # Probabilistic metrics (for CSDI and ensemble methods)
    compute_crps: bool = Field(
        default=False,
        description="Compute Continuous Ranked Probability Score (requires ensemble predictions)"
    )
    compute_calibration: bool = Field(
        default=False,
        description="Compute prediction interval calibration (requires ensemble predictions)"
    )
    compute_coverage: bool = Field(
        default=False,
        description="Compute coverage diagnostics for uncertainty quantification"
    )

    # Probabilistic metric settings
    n_samples: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of samples to generate for probabilistic metrics (CSDI)"
    )
    confidence_levels: list[float] = Field(
        default=[0.50, 0.90, 0.95],
        description="Confidence levels for prediction intervals (e.g., [0.50, 0.90, 0.95])"
    )

    @field_validator("confidence_levels")
    @classmethod
    def validate_confidence_levels(cls, v: list[float]) -> list[float]:
        """Validate confidence levels are in (0, 1)."""
        for level in v:
            if not 0.0 < level < 1.0:
                raise ValueError(f"Confidence level must be in (0, 1), got {level}")
        return v


class StratificationConfig(BaseConfig):
    """Configuration for stratified evaluation.

    Implements FR-013 (stratify by gap length, season, variable, extremes) from SPEC.md.
    """

    # Stratification dimensions
    stratify_by_gap_length: bool = Field(
        default=True,
        description="Stratify results by gap length (short, medium, long)"
    )
    stratify_by_season: bool = Field(
        default=True,
        description="Stratify results by season (winter, spring, summer, fall)"
    )
    stratify_by_variable: bool = Field(
        default=True,
        description="Report results per variable (temp, dewpoint, pressure, etc.)"
    )
    stratify_by_extremes: bool = Field(
        default=True,
        description="Stratify results by extreme values (5th/95th percentiles)"
    )

    # Gap length bins (in hours)
    gap_length_bins: list[int] = Field(
        default=[1, 6, 24, 72, 168],
        description=(
            "Gap length bin edges in hours "
            "(e.g., [1, 6, 24, 72, 168] for <6h, 6-24h, 24-72h, 72-168h, >168h)"
        )
    )

    # Extreme value thresholds (percentiles)
    extreme_lower_percentile: float = Field(
        default=5.0,
        ge=0.0,
        le=50.0,
        description="Lower percentile threshold for extreme values (default: 5th percentile)"
    )
    extreme_upper_percentile: float = Field(
        default=95.0,
        ge=50.0,
        le=100.0,
        description="Upper percentile threshold for extreme values (default: 95th percentile)"
    )

    @field_validator("gap_length_bins")
    @classmethod
    def validate_gap_length_bins(cls, v: list[int]) -> list[int]:
        """Validate gap length bins are sorted and positive."""
        if not all(x > 0 for x in v):
            raise ValueError("Gap length bins must be positive")
        if v != sorted(v):
            raise ValueError("Gap length bins must be sorted in ascending order")
        return v


class StatisticalTestConfig(BaseConfig):
    """Configuration for statistical significance testing.

    Implements FR-014 (Wilcoxon, Bonferroni, effect sizes) from SPEC.md.
    """

    # Statistical tests
    run_wilcoxon_test: bool = Field(
        default=True,
        description="Run paired Wilcoxon signed-rank test for method comparisons"
    )
    run_bonferroni_correction: bool = Field(
        default=True,
        description="Apply Bonferroni correction for multiple comparisons"
    )
    compute_cohens_d: bool = Field(
        default=True,
        description="Compute Cohen's d effect size for method differences"
    )
    compute_bootstrap_ci: bool = Field(
        default=True,
        description="Compute bootstrap confidence intervals for metrics"
    )

    # Test parameters
    alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Significance level for statistical tests"
    )
    n_bootstrap_samples: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of bootstrap samples for confidence intervals"
    )
    bootstrap_confidence_level: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Confidence level for bootstrap intervals (e.g., 0.95 for 95% CI)"
    )


class DownstreamValidationConfig(BaseConfig):
    """Configuration for downstream validation tasks.

    Implements FR-015 (degree days) and FR-016 (extreme events) from SPEC.md.
    """

    # Downstream validation tasks
    compute_degree_days: bool = Field(
        default=True,
        description="Validate imputation by computing degree days (heating/cooling/growing)"
    )
    compute_extreme_events: bool = Field(
        default=False,
        description="Validate imputation by detecting extreme events (heat waves, cold snaps)"
    )

    # Degree day thresholds (Celsius)
    heating_degree_day_base: float = Field(
        default=18.0,
        description="Base temperature for heating degree days (°C)"
    )
    cooling_degree_day_base: float = Field(
        default=18.0,
        description="Base temperature for cooling degree days (°C)"
    )
    growing_degree_day_base: float = Field(
        default=10.0,
        description="Base temperature for growing degree days (°C)"
    )

    # Extreme event thresholds
    heat_wave_threshold: float = Field(
        default=35.0,
        description="Temperature threshold for heat wave detection (°C)"
    )
    heat_wave_duration: int = Field(
        default=3,
        ge=1,
        description="Minimum consecutive days above threshold to qualify as heat wave"
    )
    cold_snap_threshold: float = Field(
        default=-10.0,
        description="Temperature threshold for cold snap detection (°C)"
    )
    cold_snap_duration: int = Field(
        default=3,
        ge=1,
        description="Minimum consecutive days below threshold to qualify as cold snap"
    )


class EvaluationConfig(BaseConfig):
    """Main evaluation configuration aggregating all sub-configurations.

    Combines metric computation, stratification, statistical testing, and downstream validation.
    """

    # Sub-configurations
    metrics: MetricConfig = Field(default_factory=MetricConfig)
    stratification: StratificationConfig = Field(default_factory=StratificationConfig)
    statistical_tests: StatisticalTestConfig = Field(default_factory=StatisticalTestConfig)
    downstream_validation: DownstreamValidationConfig = Field(
        default_factory=DownstreamValidationConfig
    )

    # Output settings
    output_format: Literal["parquet", "csv", "json"] = Field(
        default="parquet",
        description="Output format for evaluation results"
    )
    save_predictions: bool = Field(
        default=True,
        description="Save full predictions to disk (useful for re-analysis, but large files)"
    )
    save_stratified_results: bool = Field(
        default=True,
        description="Save detailed stratified results tables"
    )

    # Per-variable evaluation
    variables: list[str] = Field(
        default=[
            "temperature",
            "dew_point_temperature",
            "sea_level_pressure",
            "wind_speed",
            "wind_direction",
            "relative_humidity",
        ],
        description="Variables to evaluate (must match training data)",
    )
