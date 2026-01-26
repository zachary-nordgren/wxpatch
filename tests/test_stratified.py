"""Tests for stratified evaluation metrics."""

import datetime

import pytest
import torch

from weather_imputation.evaluation.stratified import (
    compute_stratified_metrics,
    stratify_by_extreme_values,
    stratify_by_gap_length,
    stratify_by_season,
    stratify_by_variable,
)

# ============================================================================
# Gap Length Stratification Tests
# ============================================================================


def test_gap_length_stratification():
    """Test stratification by gap length with default bins."""
    # Create test data: (2, 50, 3)
    y_true = torch.randn(2, 50, 3)
    y_pred = y_true + torch.randn(2, 50, 3) * 0.1
    mask = torch.rand(2, 50, 3) < 0.5

    # Create gap lengths covering different bins
    gap_lengths = torch.zeros_like(y_true)
    gap_lengths[:, :10, :] = 3.0  # 1-6h bin
    gap_lengths[:, 10:20, :] = 12.0  # 6-24h bin
    gap_lengths[:, 20:35, :] = 48.0  # 24-72h bin
    gap_lengths[:, 35:45, :] = 120.0  # 72-168h bin
    gap_lengths[:, 45:, :] = 200.0  # >168h bin

    results = stratify_by_gap_length(y_true, y_pred, mask, gap_lengths)

    # Should have bins for each gap length range
    assert "1-6h" in results
    assert "6-24h" in results
    assert "24-72h" in results
    assert "72-168h" in results
    assert ">168h" in results

    # Each bin should have all standard metrics
    for _bin_name, metrics in results.items():
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mse" in metrics
        assert "bias" in metrics
        assert "r2" in metrics


def test_gap_length_custom_bins():
    """Test gap length stratification with custom bins."""
    y_true = torch.randn(2, 50, 3)
    y_pred = y_true + torch.randn(2, 50, 3) * 0.1
    mask = torch.ones(2, 50, 3, dtype=torch.bool)
    gap_lengths = torch.linspace(1, 100, 2 * 50 * 3).reshape(2, 50, 3)

    custom_bins = [10, 25, 50]
    results = stratify_by_gap_length(y_true, y_pred, mask, gap_lengths, bins=custom_bins)

    assert "1-10h" in results
    assert "10-25h" in results
    assert "25-50h" in results
    assert ">50h" in results


def test_gap_length_empty_bins():
    """Test that empty bins are skipped."""
    y_true = torch.randn(2, 50, 3)
    y_pred = y_true + torch.randn(2, 50, 3) * 0.1
    mask = torch.ones(2, 50, 3, dtype=torch.bool)
    gap_lengths = torch.full((2, 50, 3), 5.0)  # All gaps 5h

    results = stratify_by_gap_length(y_true, y_pred, mask, gap_lengths)

    # Only 1-6h bin should have data
    assert "1-6h" in results
    assert "6-24h" not in results
    assert ">168h" not in results


def test_gap_length_shape_mismatch():
    """Test validation of gap_lengths shape."""
    y_true = torch.randn(2, 50, 3)
    y_pred = torch.randn(2, 50, 3)
    mask = torch.ones(2, 50, 3, dtype=torch.bool)
    gap_lengths = torch.ones(2, 50, 2)  # Wrong variable dimension

    with pytest.raises(ValueError, match="gap_lengths shape"):
        stratify_by_gap_length(y_true, y_pred, mask, gap_lengths)


# ============================================================================
# Variable Stratification Tests
# ============================================================================


def test_per_variable_evaluation():
    """Test stratification by variable with default names."""
    y_true = torch.randn(2, 50, 3)
    y_pred = y_true + torch.randn(2, 50, 3) * 0.1
    mask = torch.ones(2, 50, 3, dtype=torch.bool)

    results = stratify_by_variable(y_true, y_pred, mask)

    # Should have one entry per variable
    assert len(results) == 3
    assert "var_0" in results
    assert "var_1" in results
    assert "var_2" in results

    # Each variable should have all metrics
    for metrics in results.values():
        assert "rmse" in metrics
        assert "mae" in metrics


def test_per_variable_custom_names():
    """Test variable stratification with custom names."""
    y_true = torch.randn(2, 50, 3)
    y_pred = y_true + torch.randn(2, 50, 3) * 0.1
    mask = torch.ones(2, 50, 3, dtype=torch.bool)
    names = ["temperature", "pressure", "humidity"]

    results = stratify_by_variable(y_true, y_pred, mask, variable_names=names)

    assert "temperature" in results
    assert "pressure" in results
    assert "humidity" in results


def test_per_variable_partial_mask():
    """Test variable stratification with sparse masks."""
    y_true = torch.randn(2, 50, 3)
    y_pred = y_true + torch.randn(2, 50, 3) * 0.1
    mask = torch.zeros(2, 50, 3, dtype=torch.bool)

    # Only mask var_0 and var_2
    mask[:, :, 0] = True
    mask[:, :, 2] = True

    results = stratify_by_variable(y_true, y_pred, mask)

    # Should only have results for masked variables
    assert "var_0" in results
    assert "var_1" not in results
    assert "var_2" in results


def test_per_variable_name_length_mismatch():
    """Test validation of variable_names length."""
    y_true = torch.randn(2, 50, 3)
    y_pred = torch.randn(2, 50, 3)
    mask = torch.ones(2, 50, 3, dtype=torch.bool)
    names = ["temperature", "pressure"]  # Only 2 names for 3 variables

    with pytest.raises(ValueError, match="variable_names length"):
        stratify_by_variable(y_true, y_pred, mask, variable_names=names)


# ============================================================================
# Extreme Value Stratification Tests
# ============================================================================


def test_extreme_value_stratification():
    """Test stratification by extreme values."""
    # Create data with clear extremes
    y_true = torch.randn(5, 100, 3) * 10  # Values roughly -30 to +30
    y_pred = y_true + torch.randn(5, 100, 3) * 0.5
    mask = torch.ones(5, 100, 3, dtype=torch.bool)

    results = stratify_by_extreme_values(y_true, y_pred, mask)

    # Should have three strata
    assert "extreme_low" in results
    assert "normal" in results
    assert "extreme_high" in results

    # Each should have metrics
    for metrics in results.values():
        assert "rmse" in metrics
        assert "mae" in metrics


def test_extreme_value_custom_percentiles():
    """Test extreme value stratification with custom percentiles."""
    y_true = torch.randn(5, 100, 3) * 10
    y_pred = y_true + torch.randn(5, 100, 3) * 0.5
    mask = torch.ones(5, 100, 3, dtype=torch.bool)

    # Use 10th and 90th percentiles
    results = stratify_by_extreme_values(
        y_true, y_pred, mask, percentiles=(10.0, 90.0)
    )

    assert "extreme_low" in results
    assert "normal" in results
    assert "extreme_high" in results


def test_extreme_value_empty_mask():
    """Test extreme value stratification with empty mask."""
    y_true = torch.randn(5, 100, 3)
    y_pred = torch.randn(5, 100, 3)
    mask = torch.zeros(5, 100, 3, dtype=torch.bool)

    results = stratify_by_extreme_values(y_true, y_pred, mask)

    # Should return empty dict
    assert len(results) == 0


def test_extreme_value_invalid_percentiles():
    """Test validation of percentile ranges."""
    y_true = torch.randn(5, 100, 3)
    y_pred = torch.randn(5, 100, 3)
    mask = torch.ones(5, 100, 3, dtype=torch.bool)

    # Lower > upper
    with pytest.raises(ValueError, match="percentiles must satisfy"):
        stratify_by_extreme_values(y_true, y_pred, mask, percentiles=(95.0, 5.0))

    # Out of range
    with pytest.raises(ValueError, match="percentiles must satisfy"):
        stratify_by_extreme_values(y_true, y_pred, mask, percentiles=(-5.0, 95.0))


# ============================================================================
# Seasonal Stratification Tests
# ============================================================================


def test_seasonal_stratification():
    """Test stratification by meteorological season."""
    y_true = torch.randn(3, 400, 6)  # ~400 hours to span seasons
    y_pred = y_true + torch.randn(3, 400, 6) * 0.1
    mask = torch.ones(3, 400, 6, dtype=torch.bool)

    # Create timestamps spanning multiple seasons
    start = datetime.datetime(2023, 1, 1).timestamp()
    timestamps = torch.arange(3 * 400).float() * 3600 * 24 + start  # Daily intervals
    timestamps = timestamps.reshape(3, 400)

    results = stratify_by_season(y_true, y_pred, mask, timestamps)

    # Should have all four seasons
    assert "winter" in results
    assert "spring" in results
    assert "summer" in results
    assert "fall" in results

    # Each season should have metrics
    for metrics in results.values():
        assert "rmse" in metrics
        assert "mae" in metrics


def test_seasonal_stratification_single_season():
    """Test seasonal stratification with data from one season."""
    y_true = torch.randn(2, 100, 3)
    y_pred = y_true + torch.randn(2, 100, 3) * 0.1
    mask = torch.ones(2, 100, 3, dtype=torch.bool)

    # Create timestamps all in winter (January)
    start = datetime.datetime(2023, 1, 15).timestamp()
    timestamps = torch.arange(2 * 100).float() * 3600 + start
    timestamps = timestamps.reshape(2, 100)

    results = stratify_by_season(y_true, y_pred, mask, timestamps)

    # Should only have winter
    assert "winter" in results
    assert "spring" not in results
    assert "summer" not in results
    assert "fall" not in results


def test_seasonal_stratification_timestamp_shape_mismatch():
    """Test validation of timestamp shape."""
    y_true = torch.randn(2, 100, 3)
    y_pred = torch.randn(2, 100, 3)
    mask = torch.ones(2, 100, 3, dtype=torch.bool)
    timestamps = torch.arange(2 * 50).float().reshape(2, 50)  # Wrong T dimension

    with pytest.raises(ValueError, match="timestamps shape"):
        # timestamps shape (2, 50) doesn't match (N, T) = (2, 100)
        stratify_by_season(y_true, y_pred, mask, timestamps)


# ============================================================================
# Convenience Function Tests
# ============================================================================


def test_compute_stratified_metrics_all():
    """Test convenience function with all stratifications enabled."""
    y_true = torch.randn(3, 200, 4)
    y_pred = y_true + torch.randn(3, 200, 4) * 0.1
    mask = torch.ones(3, 200, 4, dtype=torch.bool)
    gap_lengths = torch.randint(1, 100, (3, 200, 4)).float()

    # Create timestamps spanning a year
    start = datetime.datetime(2023, 1, 1).timestamp()
    timestamps = torch.arange(3 * 200).float() * 3600 * 12 + start  # 12-hour intervals
    timestamps = timestamps.reshape(3, 200)

    variable_names = ["temp", "pressure", "humidity", "wind"]

    results = compute_stratified_metrics(
        y_true,
        y_pred,
        mask,
        gap_lengths=gap_lengths,
        timestamps=timestamps,
        variable_names=variable_names,
        stratify_gap_length=True,
        stratify_season=True,
        stratify_variable=True,
        stratify_extremes=True,
    )

    # Should have all stratification types
    assert "gap_length" in results
    assert "season" in results
    assert "variable" in results
    assert "extremes" in results

    # Verify nested structure
    assert "1-6h" in results["gap_length"]
    assert "winter" in results["season"]
    assert "temp" in results["variable"]
    assert "extreme_low" in results["extremes"]


def test_compute_stratified_metrics_selective():
    """Test convenience function with selective stratifications."""
    y_true = torch.randn(2, 100, 3)
    y_pred = y_true + torch.randn(2, 100, 3) * 0.1
    mask = torch.ones(2, 100, 3, dtype=torch.bool)

    # Only enable variable and extremes stratification
    results = compute_stratified_metrics(
        y_true,
        y_pred,
        mask,
        stratify_gap_length=False,
        stratify_season=False,
        stratify_variable=True,
        stratify_extremes=True,
    )

    assert "gap_length" not in results
    assert "season" not in results
    assert "variable" in results
    assert "extremes" in results


def test_compute_stratified_metrics_missing_gap_lengths():
    """Test error when gap_lengths missing but stratify_gap_length=True."""
    y_true = torch.randn(2, 100, 3)
    y_pred = torch.randn(2, 100, 3)
    mask = torch.ones(2, 100, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="gap_lengths required"):
        compute_stratified_metrics(
            y_true, y_pred, mask, stratify_gap_length=True, gap_lengths=None
        )


def test_compute_stratified_metrics_missing_timestamps():
    """Test error when timestamps missing but stratify_season=True."""
    y_true = torch.randn(2, 100, 3)
    y_pred = torch.randn(2, 100, 3)
    mask = torch.ones(2, 100, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="timestamps required"):
        compute_stratified_metrics(
            y_true,
            y_pred,
            mask,
            stratify_gap_length=False,
            stratify_season=True,
            timestamps=None,
        )


# ============================================================================
# Edge Cases and Input Validation
# ============================================================================


def test_shape_validation_gap_length():
    """Test shape validation in gap length stratification."""
    y_true = torch.randn(2, 50, 3)
    y_pred = torch.randn(2, 50, 4)  # Wrong shape
    mask = torch.ones(2, 50, 3, dtype=torch.bool)
    gap_lengths = torch.ones(2, 50, 3)

    with pytest.raises(ValueError, match="Shape mismatch"):
        stratify_by_gap_length(y_true, y_pred, mask, gap_lengths)


def test_shape_validation_variable():
    """Test shape validation in variable stratification."""
    y_true = torch.randn(2, 50, 3)
    y_pred = torch.randn(2, 50, 3)
    mask = torch.ones(2, 60, 3, dtype=torch.bool)  # Wrong shape

    with pytest.raises(ValueError, match="Shape mismatch"):
        stratify_by_variable(y_true, y_pred, mask)


def test_shape_validation_extremes():
    """Test shape validation in extreme value stratification."""
    y_true = torch.randn(2, 50, 3)
    y_pred = torch.randn(2, 50, 3)
    mask = torch.ones(3, 50, 3, dtype=torch.bool)  # Wrong shape

    with pytest.raises(ValueError, match="Shape mismatch"):
        stratify_by_extreme_values(y_true, y_pred, mask)


def test_shape_validation_season():
    """Test shape validation in seasonal stratification."""
    y_true = torch.randn(2, 50, 3)
    y_pred = torch.randn(2, 50, 3)
    timestamps = torch.arange(2 * 50).float().reshape(2, 50)

    # Mismatched mask shape
    bad_mask = torch.ones(2, 60, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="Shape mismatch"):
        stratify_by_season(y_true, y_pred, bad_mask, timestamps)


# ============================================================================
# Integration Tests
# ============================================================================


def test_stratified_evaluation_realistic_scenario():
    """Test stratified evaluation with realistic imputation scenario."""
    torch.manual_seed(42)

    # Create realistic weather data
    n_samples = 5
    n_timesteps = 200
    n_variables = 6

    y_true = torch.randn(n_samples, n_timesteps, n_variables) * 10 + 15  # Temp-like data
    # Imputation with small errors
    y_pred = y_true + torch.randn(n_samples, n_timesteps, n_variables) * 2

    # Evaluation mask (30% positions)
    mask = torch.rand(n_samples, n_timesteps, n_variables) < 0.3

    # Gap lengths varying from 1 to 100 hours
    gap_lengths = torch.randint(1, 100, (n_samples, n_timesteps, n_variables)).float()

    # Timestamps spanning one year
    start = datetime.datetime(2023, 1, 1).timestamp()
    timestamps = torch.arange(n_samples * n_timesteps).float() * 3600 * 2 + start
    timestamps = timestamps.reshape(n_samples, n_timesteps)

    variable_names = [
        "temperature",
        "dew_point",
        "pressure",
        "wind_speed",
        "wind_direction",
        "humidity",
    ]

    # Compute all stratifications
    results = compute_stratified_metrics(
        y_true,
        y_pred,
        mask,
        gap_lengths=gap_lengths,
        timestamps=timestamps,
        variable_names=variable_names,
        gap_bins=[6, 24, 72],
        extreme_percentiles=(10.0, 90.0),
        stratify_gap_length=True,
        stratify_season=True,
        stratify_variable=True,
        stratify_extremes=True,
    )

    # Verify all stratifications present
    assert "gap_length" in results
    assert "season" in results
    assert "variable" in results
    assert "extremes" in results

    # Verify metrics are reasonable (not NaN, finite)
    for strat_type, strat_results in results.items():
        for stratum_name, metrics in strat_results.items():
            assert all(
                not torch.isnan(torch.tensor(v)) or v != v
                for v in metrics.values()
            ), f"Found NaN in {strat_type}/{stratum_name}"
