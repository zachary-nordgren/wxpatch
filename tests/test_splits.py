"""Tests for train/val/test splitting strategies."""

from datetime import datetime, timedelta

import polars as pl
import pytest
import torch

from weather_imputation.data.splits import (
    create_split,
    create_temporal_mask,
    split_by_temporal_mask,
    split_hybrid,
    split_simulated,
    split_spatial,
    split_temporal,
)


@pytest.fixture
def sample_metadata() -> pl.DataFrame:
    """Create sample station metadata for testing."""
    return pl.DataFrame(
        {
            "station_id": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "station_name": [f"Station {i}" for i in range(10)],
            "latitude": [30.0 + i for i in range(10)],
            "longitude": [-100.0 - i for i in range(10)],
            "temperature_completeness_pct": [70.0 + i for i in range(10)],
        }
    )


@pytest.fixture
def temporal_metadata() -> pl.DataFrame:
    """Create metadata with temporal information."""
    base_date = datetime(2020, 1, 1)
    return pl.DataFrame(
        {
            "station_id": ["A", "B", "C", "D", "E"],
            "first_observation": [base_date] * 5,
            "last_observation": [base_date + timedelta(days=365 * 3)] * 5,
        }
    )


# ======================
# Spatial Split Tests
# ======================


def test_split_spatial_basic(sample_metadata):
    """Test basic spatial split functionality."""
    train_ids, val_ids, test_ids = split_spatial(
        sample_metadata,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    # Check that all stations are assigned
    all_ids = set(train_ids + val_ids + test_ids)
    assert len(all_ids) == 10
    assert all_ids == set(sample_metadata["station_id"].to_list())

    # Check approximate proportions (allow ±1 due to rounding)
    assert 6 <= len(train_ids) <= 8  # ~70%
    assert 0 <= len(val_ids) <= 3  # ~15%
    assert 0 <= len(test_ids) <= 3  # ~15%


def test_split_spatial_no_overlap(sample_metadata):
    """Test that spatial splits don't overlap."""
    train_ids, val_ids, test_ids = split_spatial(sample_metadata, seed=42)

    assert not set(train_ids) & set(val_ids), "Train and val should not overlap"
    assert not set(train_ids) & set(test_ids), "Train and test should not overlap"
    assert not set(val_ids) & set(test_ids), "Val and test should not overlap"


def test_split_spatial_reproducible(sample_metadata):
    """Test that spatial split is reproducible with same seed."""
    train1, val1, test1 = split_spatial(sample_metadata, seed=42)
    train2, val2, test2 = split_spatial(sample_metadata, seed=42)

    assert train1 == train2
    assert val1 == val2
    assert test1 == test2


def test_split_spatial_different_seeds(sample_metadata):
    """Test that different seeds produce different splits."""
    train1, val1, test1 = split_spatial(sample_metadata, seed=42)
    train2, val2, test2 = split_spatial(sample_metadata, seed=123)

    # At least one split should differ
    assert train1 != train2 or val1 != val2 or test1 != test2


def test_split_spatial_invalid_ratios(sample_metadata):
    """Test that invalid ratios raise ValueError."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        split_spatial(sample_metadata, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_split_spatial_empty_metadata():
    """Test that empty metadata raises ValueError."""
    empty_metadata = pl.DataFrame({"station_id": []})
    with pytest.raises(ValueError, match="empty"):
        split_spatial(empty_metadata)


# ======================
# Temporal Split Tests
# ======================


def test_split_temporal_basic(temporal_metadata):
    """Test basic temporal split functionality."""
    train_ids, val_ids, test_ids = split_temporal(
        temporal_metadata,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    # All stations should be in all splits for temporal strategy
    assert train_ids == val_ids == test_ids
    assert set(train_ids) == set(temporal_metadata["station_id"].to_list())


def test_split_temporal_missing_columns(sample_metadata):
    """Test that missing temporal columns raises ValueError."""
    with pytest.raises(ValueError, match="Missing required columns"):
        split_temporal(sample_metadata)


def test_split_temporal_invalid_ratios(temporal_metadata):
    """Test that invalid ratios raise ValueError."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        split_temporal(
            temporal_metadata, train_ratio=0.6, val_ratio=0.2, test_ratio=0.3
        )


# ======================
# Hybrid Split Tests
# ======================


def test_split_hybrid_basic(sample_metadata):
    """Test basic hybrid split functionality."""
    train_ids, val_ids, test_ids = split_hybrid(
        sample_metadata,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        spatial_weight=0.5,
        seed=42,
    )

    # Check that all stations are assigned
    all_ids = set(train_ids + val_ids + test_ids)
    assert len(all_ids) == 10

    # Check approximate proportions
    assert len(train_ids) >= 5  # At least half for training
    assert len(test_ids) >= 1  # Some stations held out


def test_split_hybrid_spatial_weight_zero(sample_metadata):
    """Test hybrid split with spatial_weight=0 (all temporal)."""
    train_ids, val_ids, test_ids = split_hybrid(
        sample_metadata,
        spatial_weight=0.0,
        seed=42,
    )

    # With spatial_weight=0, should behave like pure spatial split
    # (no spatial holdout, but stations still split)
    assert len(train_ids) + len(val_ids) + len(test_ids) == 10


def test_split_hybrid_spatial_weight_one(sample_metadata):
    """Test hybrid split with spatial_weight=1.0 (all spatial)."""
    train_ids, val_ids, test_ids = split_hybrid(
        sample_metadata,
        spatial_weight=1.0,
        seed=42,
    )

    # With spatial_weight=1.0, test set is entirely spatial holdout
    assert len(test_ids) >= 1


def test_split_hybrid_invalid_spatial_weight(sample_metadata):
    """Test that invalid spatial_weight raises ValueError."""
    with pytest.raises(ValueError, match="spatial_weight must be"):
        split_hybrid(sample_metadata, spatial_weight=1.5)

    with pytest.raises(ValueError, match="spatial_weight must be"):
        split_hybrid(sample_metadata, spatial_weight=-0.1)


def test_split_hybrid_invalid_ratios(sample_metadata):
    """Test that invalid ratios raise ValueError."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        split_hybrid(
            sample_metadata, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3
        )


# ======================
# Simulated Split Tests (Strategy D)
# ======================


def test_strategy_d_split(sample_metadata):
    """Test Strategy D: simulated masks within signals (main test case for TASK-013)."""
    train_ids, val_ids, test_ids = split_simulated(
        sample_metadata,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    # Strategy D: All stations in all splits
    assert train_ids == val_ids == test_ids
    assert set(train_ids) == set(sample_metadata["station_id"].to_list())
    assert len(train_ids) == 10


def test_split_simulated_all_stations_included(sample_metadata):
    """Test that all stations are included in all splits."""
    train_ids, val_ids, test_ids = split_simulated(sample_metadata, seed=42)

    expected_ids = set(sample_metadata["station_id"].to_list())
    assert set(train_ids) == expected_ids
    assert set(val_ids) == expected_ids
    assert set(test_ids) == expected_ids


def test_split_simulated_invalid_ratios(sample_metadata):
    """Test that invalid ratios raise ValueError."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        split_simulated(
            sample_metadata, train_ratio=0.6, val_ratio=0.3, test_ratio=0.2
        )


def test_split_simulated_empty_metadata():
    """Test that empty metadata raises ValueError."""
    empty_metadata = pl.DataFrame({"station_id": []})
    with pytest.raises(ValueError, match="empty"):
        split_simulated(empty_metadata)


def test_split_simulated_different_ratios(sample_metadata):
    """Test that different ratios still return all stations."""
    # Ratios don't affect which stations are returned (all are returned)
    # They're used later during masking
    train_ids, val_ids, test_ids = split_simulated(
        sample_metadata,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    assert train_ids == val_ids == test_ids
    assert len(train_ids) == 10


# ======================
# create_split Dispatcher Tests
# ======================


def test_create_split_spatial(sample_metadata):
    """Test create_split dispatcher with spatial strategy."""
    train_ids, val_ids, test_ids = create_split(
        sample_metadata, strategy="spatial", seed=42
    )

    # Should behave like split_spatial
    assert len(set(train_ids + val_ids + test_ids)) == 10
    assert not set(train_ids) & set(val_ids)


def test_create_split_temporal(temporal_metadata):
    """Test create_split dispatcher with temporal strategy."""
    train_ids, val_ids, test_ids = create_split(
        temporal_metadata, strategy="temporal"
    )

    # Should behave like split_temporal
    assert train_ids == val_ids == test_ids


def test_create_split_hybrid(sample_metadata):
    """Test create_split dispatcher with hybrid strategy."""
    train_ids, val_ids, test_ids = create_split(
        sample_metadata,
        strategy="hybrid",
        spatial_weight=0.5,
        seed=42,
    )

    # Should behave like split_hybrid
    assert len(set(train_ids + val_ids + test_ids)) == 10


def test_create_split_simulated(sample_metadata):
    """Test create_split dispatcher with simulated strategy (Strategy D)."""
    train_ids, val_ids, test_ids = create_split(
        sample_metadata, strategy="simulated", seed=42
    )

    # Should behave like split_simulated
    assert train_ids == val_ids == test_ids
    assert len(train_ids) == 10


def test_create_split_unknown_strategy(sample_metadata):
    """Test that unknown strategy raises ValueError."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        create_split(sample_metadata, strategy="invalid_strategy")


def test_create_split_default_strategy(sample_metadata):
    """Test that default strategy is 'simulated' (Strategy D)."""
    train_ids, val_ids, test_ids = create_split(sample_metadata, seed=42)

    # Default should be Strategy D (simulated)
    assert train_ids == val_ids == test_ids


# ======================
# Temporal Mask Tests
# ======================


def test_create_temporal_mask_basic():
    """Test basic temporal mask creation."""
    mask = create_temporal_mask(100, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert mask.shape == (100,)
    assert mask.dtype == torch.long

    # Check split sizes
    assert (mask == 0).sum() == 70  # Train
    assert (mask == 1).sum() == 15  # Val
    assert (mask == 2).sum() == 15  # Test


def test_create_temporal_mask_proportions():
    """Test that temporal mask respects ratios."""
    n = 1000
    mask = create_temporal_mask(n, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    n_train = (mask == 0).sum().item()
    n_val = (mask == 1).sum().item()
    n_test = (mask == 2).sum().item()

    assert n_train == 600
    assert n_val == 200
    assert n_test == 200


def test_create_temporal_mask_contiguous():
    """Test that temporal mask creates contiguous segments."""
    mask = create_temporal_mask(100)

    # Train should be first
    train_end = (mask == 0).sum().item()
    assert (mask[:train_end] == 0).all()

    # Val should be next
    val_start = train_end
    val_end = val_start + (mask == 1).sum().item()
    assert (mask[val_start:val_end] == 1).all()

    # Test should be last
    assert (mask[val_end:] == 2).all()


def test_create_temporal_mask_invalid_ratios():
    """Test that invalid ratios raise ValueError."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        create_temporal_mask(100, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_create_temporal_mask_invalid_length():
    """Test that invalid length raises ValueError."""
    with pytest.raises(ValueError, match="must be >= 1"):
        create_temporal_mask(0)


def test_create_temporal_mask_small_length():
    """Test temporal mask with very small length."""
    mask = create_temporal_mask(3, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert mask.shape == (3,)
    # With n=3: train gets 2, val gets 0, test gets 1
    assert (mask == 0).sum() == 2
    assert (mask == 2).sum() == 1


# ======================
# Split by Temporal Mask Tests
# ======================


def test_split_by_temporal_mask_2d():
    """Test splitting 2D data (T, V) by temporal mask."""
    data = torch.randn(100, 6)  # 100 timesteps, 6 variables
    mask = create_temporal_mask(100, 0.7, 0.15, 0.15)

    train_data = split_by_temporal_mask(data, mask, "train")
    val_data = split_by_temporal_mask(data, mask, "val")
    test_data = split_by_temporal_mask(data, mask, "test")

    assert train_data.shape == (70, 6)
    assert val_data.shape == (15, 6)
    assert test_data.shape == (15, 6)


def test_split_by_temporal_mask_3d():
    """Test splitting 3D data (N, T, V) by temporal mask."""
    data = torch.randn(32, 168, 6)  # 32 samples, 168 timesteps, 6 variables
    mask = create_temporal_mask(168, 0.7, 0.15, 0.15)

    train_data = split_by_temporal_mask(data, mask, "train")
    val_data = split_by_temporal_mask(data, mask, "val")
    test_data = split_by_temporal_mask(data, mask, "test")

    assert train_data.shape == (32, 117, 6)  # 70% of 168
    assert val_data.shape == (32, 25, 6)  # 15% of 168
    assert test_data.shape == (32, 26, 6)  # 15% of 168


def test_split_by_temporal_mask_preserves_values():
    """Test that split preserves actual data values."""
    data = torch.arange(100).float().view(100, 1)
    mask = create_temporal_mask(100, 0.7, 0.15, 0.15)

    train_data = split_by_temporal_mask(data, mask, "train")

    # Check that train data contains first 70 values
    assert torch.allclose(train_data.squeeze(), torch.arange(70).float())


def test_split_by_temporal_mask_unknown_split():
    """Test that unknown split name raises ValueError."""
    data = torch.randn(100, 6)
    mask = create_temporal_mask(100)

    with pytest.raises(ValueError, match="Unknown split"):
        split_by_temporal_mask(data, mask, "invalid")


def test_split_by_temporal_mask_incompatible_shapes():
    """Test behavior with incompatible shapes (currently not validated)."""
    # Note: Current implementation doesn't validate shape compatibility
    # This is a placeholder for future improvement
    # For now, we just document the expected behavior
    pass  # Future: add shape validation and test it


def test_split_by_temporal_mask_1d_raises():
    """Test that 1D data raises ValueError."""
    data = torch.randn(100)
    mask = create_temporal_mask(100)

    with pytest.raises(ValueError, match="must be 2D or 3D"):
        split_by_temporal_mask(data, mask, "train")


# ======================
# Integration Tests
# ======================


def test_strategy_d_workflow(sample_metadata):
    """Test complete Strategy D workflow: split + temporal mask + data split."""
    # Step 1: Create simulated split (returns all stations)
    train_ids, val_ids, test_ids = split_simulated(sample_metadata, seed=42)
    assert train_ids == val_ids == test_ids

    # Step 2: Create temporal mask for time series
    mask = create_temporal_mask(168, 0.7, 0.15, 0.15)

    # Step 3: Create dummy data for one station
    station_data = torch.randn(168, 6)  # 1 week, 6 variables

    # Step 4: Extract train/val/test portions
    train_data = split_by_temporal_mask(station_data, mask, "train")
    val_data = split_by_temporal_mask(station_data, mask, "val")
    test_data = split_by_temporal_mask(station_data, mask, "test")

    # Verify sizes
    assert train_data.shape == (117, 6)  # ~70%
    assert val_data.shape == (25, 6)  # ~15%
    assert test_data.shape == (26, 6)  # ~15%


def test_all_strategies_produce_valid_splits(sample_metadata, temporal_metadata):
    """Test that all strategies produce valid, non-empty splits."""
    strategies_and_metadata = [
        ("spatial", sample_metadata),
        ("temporal", temporal_metadata),
        ("hybrid", sample_metadata),
        ("simulated", sample_metadata),
    ]

    for strategy, metadata in strategies_and_metadata:
        train_ids, val_ids, test_ids = create_split(
            metadata, strategy=strategy, seed=42
        )

        # All splits should be non-empty for these dataset sizes
        assert len(train_ids) > 0, f"{strategy} produced empty train set"
        # Val and test might be empty for small datasets with some strategies
        # but at least train should always have data

        # All IDs should be strings
        assert all(isinstance(id, str) for id in train_ids)
        assert all(isinstance(id, str) for id in val_ids)
        assert all(isinstance(id, str) for id in test_ids)
