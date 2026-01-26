"""Tests for normalization utilities."""

import pytest
import torch

from weather_imputation.data.normalization import (
    NormalizationStats,
    Normalizer,
    normalize_variables,
)

# ========================================
# Fixtures
# ========================================


@pytest.fixture
def sample_data():
    """Sample time series data (N=10, T=24, V=3)."""
    torch.manual_seed(42)
    # Create data with different scales for each variable
    # Variable 0: temperature-like (mean=15, std=5)
    var0 = torch.randn(10, 24) * 5 + 15
    # Variable 1: pressure-like (mean=1013, std=10)
    var1 = torch.randn(10, 24) * 10 + 1013
    # Variable 2: humidity-like (0-100 range)
    var2 = torch.rand(10, 24) * 100

    data = torch.stack([var0, var1, var2], dim=2)  # (10, 24, 3)
    return data


@pytest.fixture
def sample_mask(sample_data):
    """Sample mask with ~20% missing values."""
    torch.manual_seed(42)
    mask = torch.rand(sample_data.shape) > 0.2  # True = observed
    return mask


# ========================================
# NormalizationStats Tests
# ========================================


def test_normalization_stats_creation():
    """Test creating NormalizationStats."""
    stats = NormalizationStats(
        mean=15.0, std=5.0, min_val=0.0, max_val=30.0, n_observed=1000
    )
    assert stats.mean == 15.0
    assert stats.std == 5.0
    assert stats.min_val == 0.0
    assert stats.max_val == 30.0
    assert stats.n_observed == 1000


# ========================================
# Normalizer Initialization Tests
# ========================================


def test_normalizer_initialization():
    """Test Normalizer initialization with different methods."""
    # Default (zscore)
    norm = Normalizer()
    assert norm.method == "zscore"
    assert norm.clip_outliers is False
    assert not norm.is_fitted()

    # Min-max
    norm = Normalizer(method="minmax")
    assert norm.method == "minmax"
    assert not norm.is_fitted()

    # None (no normalization)
    norm = Normalizer(method="none")
    assert norm.method == "none"
    assert not norm.is_fitted()

    # With outlier clipping
    norm = Normalizer(method="zscore", clip_outliers=True)
    assert norm.clip_outliers is True


# ========================================
# Normalizer Fit Tests
# ========================================


def test_normalizer_fit_zscore(sample_data, sample_mask):
    """Test fitting normalizer with z-score method."""
    normalizer = Normalizer(method="zscore")
    normalizer.fit(sample_data, sample_mask)

    assert normalizer.is_fitted()
    assert len(normalizer.stats) == 3  # 3 variables

    # Check stats for first variable (temperature-like)
    stats0 = normalizer.get_stats(0)
    assert stats0 is not None
    assert 10 < stats0.mean < 20  # Should be around 15
    assert 3 < stats0.std < 7  # Should be around 5
    assert stats0.n_observed > 0


def test_normalizer_fit_minmax(sample_data, sample_mask):
    """Test fitting normalizer with min-max method."""
    normalizer = Normalizer(method="minmax")
    normalizer.fit(sample_data, sample_mask)

    assert normalizer.is_fitted()
    assert len(normalizer.stats) == 3

    # Check stats for first variable
    stats0 = normalizer.get_stats(0)
    assert stats0 is not None
    assert stats0.min_val < stats0.max_val
    assert stats0.n_observed > 0


def test_normalizer_fit_shape_validation(sample_data, sample_mask):
    """Test that fit validates input shapes."""
    normalizer = Normalizer()

    # Mismatched shapes
    with pytest.raises(ValueError, match="must match mask shape"):
        normalizer.fit(sample_data, sample_mask[:, :5, :])

    # Wrong number of dimensions
    with pytest.raises(ValueError, match="must be 3D"):
        normalizer.fit(sample_data[:, :, 0], sample_mask[:, :, 0])


def test_normalizer_fit_no_observed_values():
    """Test that fit raises error when variable has no observed values."""
    data = torch.randn(10, 24, 3)
    mask = torch.ones_like(data, dtype=torch.bool)
    mask[:, :, 1] = False  # Variable 1 has no observed values

    normalizer = Normalizer()
    with pytest.raises(ValueError, match="has no observed values"):
        normalizer.fit(data, mask)


def test_normalizer_fit_constant_variable():
    """Test that fit handles constant variables (std=0 or min=max)."""
    # Create data where variable 1 is constant
    data = torch.randn(10, 24, 3)
    data[:, :, 1] = 5.0  # Constant value
    mask = torch.ones_like(data, dtype=torch.bool)

    # Should not raise error
    normalizer = Normalizer(method="zscore")
    normalizer.fit(data, mask)

    stats1 = normalizer.get_stats(1)
    assert stats1.mean == 5.0
    assert stats1.std == 1.0  # Should be set to 1.0 to avoid division by zero

    # Min-max normalization
    normalizer = Normalizer(method="minmax")
    normalizer.fit(data, mask)

    stats1 = normalizer.get_stats(1)
    assert stats1.min_val == 5.0
    assert stats1.max_val == 6.0  # Should be min + 1.0 to avoid division by zero


# ========================================
# Normalizer Transform Tests
# ========================================


def test_normalizer_transform_zscore(sample_data, sample_mask):
    """Test z-score transformation."""
    normalizer = Normalizer(method="zscore")
    normalizer.fit(sample_data, sample_mask)
    normalized = normalizer.transform(sample_data, sample_mask)

    # Check shape preserved
    assert normalized.shape == sample_data.shape

    # Check that observed values are normalized (mean~0, std~1)
    for v in range(3):
        var_mask = sample_mask[:, :, v]
        var_normalized = normalized[:, :, v][var_mask]
        mean = var_normalized.mean()
        std = var_normalized.std()

        # Should be approximately standard normal
        assert abs(mean) < 0.5, f"Variable {v} mean {mean} not close to 0"
        assert abs(std - 1.0) < 0.5, f"Variable {v} std {std} not close to 1"

    # Check that missing values are unchanged (not normalized)
    missing_mask = ~sample_mask
    assert torch.equal(
        normalized[missing_mask], sample_data[missing_mask]
    ), "Missing values should not be normalized"


def test_normalizer_transform_minmax(sample_data, sample_mask):
    """Test min-max transformation."""
    normalizer = Normalizer(method="minmax")
    normalizer.fit(sample_data, sample_mask)
    normalized = normalizer.transform(sample_data, sample_mask)

    # Check shape preserved
    assert normalized.shape == sample_data.shape

    # Check that observed values are in [0, 1] range
    for v in range(3):
        var_mask = sample_mask[:, :, v]
        var_normalized = normalized[:, :, v][var_mask]

        # Should be in [0, 1] range (with small tolerance for floating point)
        assert var_normalized.min() >= -0.01, f"Variable {v} has values < 0"
        assert var_normalized.max() <= 1.01, f"Variable {v} has values > 1"


def test_normalizer_transform_none(sample_data, sample_mask):
    """Test that 'none' method returns unchanged data."""
    normalizer = Normalizer(method="none")
    normalizer.fit(sample_data, sample_mask)
    normalized = normalizer.transform(sample_data, sample_mask)

    # Should be identical (but different object)
    assert torch.allclose(normalized, sample_data)
    assert normalized is not sample_data  # Should be a clone


def test_normalizer_transform_clip_outliers():
    """Test outlier clipping in z-score normalization."""
    # Create data with extreme outliers
    data = torch.randn(10, 24, 2)
    data[0, 0, 0] = 100.0  # Extreme outlier
    data[0, 1, 0] = -100.0  # Extreme outlier
    mask = torch.ones_like(data, dtype=torch.bool)

    normalizer = Normalizer(method="zscore", clip_outliers=True)
    normalizer.fit(data, mask)
    normalized = normalizer.transform(data, mask)

    # Outliers should be clipped to [-5, 5]
    assert normalized[:, :, 0].max() <= 5.0
    assert normalized[:, :, 0].min() >= -5.0


def test_normalizer_transform_not_fitted():
    """Test that transform raises error if not fitted."""
    normalizer = Normalizer()
    data = torch.randn(10, 24, 3)
    mask = torch.ones_like(data, dtype=torch.bool)

    with pytest.raises(ValueError, match="must be fitted"):
        normalizer.transform(data, mask)


def test_normalizer_transform_shape_mismatch(sample_data, sample_mask):
    """Test that transform validates variable count."""
    normalizer = Normalizer()
    normalizer.fit(sample_data, sample_mask)

    # Wrong number of variables
    wrong_data = torch.randn(10, 24, 5)  # 5 variables instead of 3
    wrong_mask = torch.ones_like(wrong_data, dtype=torch.bool)

    with pytest.raises(ValueError, match="was fitted with"):
        normalizer.transform(wrong_data, wrong_mask)


# ========================================
# Normalizer Inverse Transform Tests
# ========================================


def test_normalizer_inverse_transform_zscore(sample_data, sample_mask):
    """Test inverse z-score transformation."""
    normalizer = Normalizer(method="zscore")
    normalizer.fit(sample_data, sample_mask)
    normalized = normalizer.transform(sample_data, sample_mask)
    reconstructed = normalizer.inverse_transform(normalized)

    # Should reconstruct original values (for observed positions)
    observed_original = sample_data[sample_mask]
    observed_reconstructed = reconstructed[sample_mask]

    assert torch.allclose(
        observed_original, observed_reconstructed, atol=1e-5
    ), "Inverse transform should reconstruct original values"


def test_normalizer_inverse_transform_minmax(sample_data, sample_mask):
    """Test inverse min-max transformation."""
    normalizer = Normalizer(method="minmax")
    normalizer.fit(sample_data, sample_mask)
    normalized = normalizer.transform(sample_data, sample_mask)
    reconstructed = normalizer.inverse_transform(normalized)

    # Should reconstruct original values (for observed positions)
    observed_original = sample_data[sample_mask]
    observed_reconstructed = reconstructed[sample_mask]

    assert torch.allclose(
        observed_original, observed_reconstructed, atol=1e-5
    ), "Inverse transform should reconstruct original values"


def test_normalizer_inverse_transform_none(sample_data, sample_mask):
    """Test that 'none' method returns unchanged data for inverse."""
    normalizer = Normalizer(method="none")
    normalizer.fit(sample_data, sample_mask)
    normalized = normalizer.transform(sample_data, sample_mask)
    reconstructed = normalizer.inverse_transform(normalized)

    assert torch.allclose(reconstructed, sample_data)


def test_normalizer_inverse_transform_not_fitted():
    """Test that inverse_transform raises error if not fitted."""
    normalizer = Normalizer()
    data = torch.randn(10, 24, 3)

    with pytest.raises(ValueError, match="must be fitted"):
        normalizer.inverse_transform(data)


# ========================================
# normalize_variables Convenience Function Tests
# ========================================


def test_normalize_variables(sample_data, sample_mask):
    """Test normalize_variables convenience function."""
    normalized, normalizer = normalize_variables(
        sample_data, sample_mask, method="zscore"
    )

    # Should return normalized data and fitted normalizer
    assert normalized.shape == sample_data.shape
    assert normalizer.is_fitted()

    # Should be equivalent to manual fit + transform
    manual_normalizer = Normalizer(method="zscore")
    manual_normalizer.fit(sample_data, sample_mask)
    manual_normalized = manual_normalizer.transform(sample_data, sample_mask)

    assert torch.allclose(normalized, manual_normalized)


def test_normalize_variables_with_clipping(sample_data, sample_mask):
    """Test normalize_variables with outlier clipping."""
    normalized, normalizer = normalize_variables(
        sample_data, sample_mask, method="zscore", clip_outliers=True
    )

    assert normalizer.clip_outliers is True
    assert normalizer.is_fitted()


# ========================================
# Edge Cases and Integration Tests
# ========================================


def test_normalizer_with_all_observed():
    """Test normalizer when all values are observed (no missing)."""
    data = torch.randn(10, 24, 3)
    mask = torch.ones_like(data, dtype=torch.bool)  # All observed

    normalizer = Normalizer(method="zscore")
    normalizer.fit(data, mask)
    normalized = normalizer.transform(data, mask)
    reconstructed = normalizer.inverse_transform(normalized)

    assert torch.allclose(data, reconstructed, atol=1e-5)


def test_normalizer_with_mostly_missing():
    """Test normalizer when most values are missing."""
    data = torch.randn(10, 24, 3)
    mask = torch.rand_like(data) > 0.9  # Only ~10% observed

    normalizer = Normalizer(method="zscore")
    normalizer.fit(data, mask)
    normalized = normalizer.transform(data, mask)

    # Should still work with limited observed data
    assert normalized.shape == data.shape
    assert normalizer.is_fitted()


def test_normalizer_single_sample():
    """Test normalizer with single sample (N=1)."""
    data = torch.randn(1, 24, 3)
    mask = torch.ones_like(data, dtype=torch.bool)

    normalizer = Normalizer(method="zscore")
    normalizer.fit(data, mask)
    normalized = normalizer.transform(data, mask)

    assert normalized.shape == data.shape


def test_normalizer_reproducibility():
    """Test that normalization is deterministic."""
    torch.manual_seed(42)
    data1 = torch.randn(10, 24, 3)
    mask1 = torch.ones_like(data1, dtype=torch.bool)

    torch.manual_seed(42)
    data2 = torch.randn(10, 24, 3)
    mask2 = torch.ones_like(data2, dtype=torch.bool)

    normalizer1 = Normalizer(method="zscore")
    normalizer1.fit(data1, mask1)
    normalized1 = normalizer1.transform(data1, mask1)

    normalizer2 = Normalizer(method="zscore")
    normalizer2.fit(data2, mask2)
    normalized2 = normalizer2.transform(data2, mask2)

    assert torch.allclose(normalized1, normalized2)
