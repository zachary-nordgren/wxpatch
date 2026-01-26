"""Tests for PyTorch Dataset for time series imputation."""

import pytest
import torch

from weather_imputation.data.dataset import TimeSeriesImputationDataset

# ========================================
# Fixtures
# ========================================


@pytest.fixture
def sample_data():
    """Sample time series data (N=5, T=100, V=3)."""
    torch.manual_seed(42)
    return torch.randn(5, 100, 3)


@pytest.fixture
def sample_mask(sample_data):
    """Sample mask with all values observed."""
    return torch.ones(sample_data.shape, dtype=torch.bool)


@pytest.fixture
def sample_timestamps(sample_data):
    """Sample timestamps."""
    n_samples, seq_length = sample_data.shape[:2]
    # Create timestamps starting from 0 with 1-hour increments
    base_timestamps = torch.arange(seq_length, dtype=torch.long)
    return base_timestamps.unsqueeze(0).expand(n_samples, -1)


@pytest.fixture
def sample_station_features():
    """Sample station features (N=5, F=3)."""
    torch.manual_seed(42)
    # lat, lon, elevation (normalized)
    return torch.randn(5, 3)


# ========================================
# Initialization Tests
# ========================================


def test_dataset_initialization(sample_data, sample_mask, sample_timestamps):
    """Test basic dataset initialization."""
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
    )

    assert dataset.n_samples == 5
    assert dataset.seq_length == 100
    assert dataset.n_variables == 3
    assert dataset.window_size == 20
    assert dataset.stride == 10
    assert dataset.windows_per_sample == 9  # (100 - 20) // 10 + 1
    assert len(dataset) == 45  # 5 samples * 9 windows


def test_dataset_with_station_features(
    sample_data, sample_mask, sample_timestamps, sample_station_features
):
    """Test dataset initialization with station features."""
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        station_features=sample_station_features,
        window_size=20,
        stride=10,
    )

    assert dataset.station_features is not None
    assert dataset.station_features.shape == (5, 3)


def test_dataset_validation_data_shape(sample_mask, sample_timestamps):
    """Test validation of data shape."""
    # 2D data (should be 3D)
    with pytest.raises(ValueError, match="data must be 3D"):
        TimeSeriesImputationDataset(
            data=torch.randn(5, 100),
            mask=sample_mask,
            timestamps=sample_timestamps,
        )

    # 4D data
    with pytest.raises(ValueError, match="data must be 3D"):
        TimeSeriesImputationDataset(
            data=torch.randn(5, 100, 3, 2),
            mask=sample_mask,
            timestamps=sample_timestamps,
        )


def test_dataset_validation_mask_shape(sample_data, sample_timestamps):
    """Test validation of mask shape."""
    with pytest.raises(ValueError, match="mask shape .* must match data shape"):
        TimeSeriesImputationDataset(
            data=sample_data,
            mask=torch.ones(5, 50, 3, dtype=torch.bool),  # Wrong T dimension
            timestamps=sample_timestamps,
        )


def test_dataset_validation_timestamps_shape(sample_data, sample_mask):
    """Test validation of timestamps shape."""
    with pytest.raises(ValueError, match="timestamps shape .* must match data shape"):
        TimeSeriesImputationDataset(
            data=sample_data,
            mask=sample_mask,
            timestamps=torch.arange(50).unsqueeze(0).expand(5, -1),  # Wrong T
        )


def test_dataset_validation_station_features_shape(
    sample_data, sample_mask, sample_timestamps
):
    """Test validation of station features shape."""
    with pytest.raises(ValueError, match="station_features must have .* samples"):
        TimeSeriesImputationDataset(
            data=sample_data,
            mask=sample_mask,
            timestamps=sample_timestamps,
            station_features=torch.randn(3, 3),  # Wrong N dimension
        )


def test_dataset_validation_window_size(sample_data, sample_mask, sample_timestamps):
    """Test validation of window size."""
    with pytest.raises(ValueError, match="window_size .* cannot exceed sequence length"):
        TimeSeriesImputationDataset(
            data=sample_data,
            mask=sample_mask,
            timestamps=sample_timestamps,
            window_size=200,  # Larger than seq_length=100
        )


def test_dataset_validation_stride(sample_data, sample_mask, sample_timestamps):
    """Test validation of stride."""
    with pytest.raises(ValueError, match="stride must be >= 1"):
        TimeSeriesImputationDataset(
            data=sample_data,
            mask=sample_mask,
            timestamps=sample_timestamps,
            stride=0,
        )


# ========================================
# Window Computation Tests
# ========================================


def test_windows_per_sample_computation(sample_data, sample_mask, sample_timestamps):
    """Test computation of windows per sample."""
    # window_size=20, stride=10, seq_length=100
    # windows_per_sample = (100 - 20) // 10 + 1 = 9
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
    )
    assert dataset.windows_per_sample == 9

    # window_size=50, stride=25, seq_length=100
    # windows_per_sample = (100 - 50) // 25 + 1 = 3
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=50,
        stride=25,
    )
    assert dataset.windows_per_sample == 3

    # window_size=100, stride=1, seq_length=100
    # windows_per_sample = (100 - 100) // 1 + 1 = 1
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=100,
        stride=1,
    )
    assert dataset.windows_per_sample == 1


def test_total_windows_computation(sample_data, sample_mask, sample_timestamps):
    """Test computation of total windows."""
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
    )

    # 5 samples * 9 windows per sample = 45
    assert len(dataset) == 45


# ========================================
# __getitem__ Tests
# ========================================


def test_getitem_basic(sample_data, sample_mask, sample_timestamps):
    """Test basic __getitem__ functionality."""
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
        apply_synthetic_mask=False,  # No synthetic masking
    )

    sample = dataset[0]

    # Check keys
    assert "observed" in sample
    assert "mask" in sample
    assert "target" in sample
    assert "timestamps" in sample
    assert "sample_idx" in sample
    assert "window_start" in sample

    # Check shapes
    assert sample["observed"].shape == (20, 3)  # (T, V)
    assert sample["mask"].shape == (20, 3)
    assert sample["target"].shape == (20, 3)
    assert sample["timestamps"].shape == (20,)

    # Check types
    assert sample["observed"].dtype == torch.float32
    assert sample["mask"].dtype == torch.bool
    assert sample["target"].dtype == torch.float32
    assert sample["timestamps"].dtype == torch.long


def test_getitem_window_extraction(sample_data, sample_mask, sample_timestamps):
    """Test that windows are extracted correctly."""
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
        apply_synthetic_mask=False,
    )

    # First window of first sample (idx=0)
    sample = dataset[0]
    assert sample["sample_idx"].item() == 0
    assert sample["window_start"].item() == 0
    torch.testing.assert_close(
        sample["target"], sample_data[0, 0:20, :]
    )

    # Second window of first sample (idx=1)
    sample = dataset[1]
    assert sample["sample_idx"].item() == 0
    assert sample["window_start"].item() == 10
    torch.testing.assert_close(
        sample["target"], sample_data[0, 10:30, :]
    )

    # First window of second sample (idx=9)
    sample = dataset[9]
    assert sample["sample_idx"].item() == 1
    assert sample["window_start"].item() == 0
    torch.testing.assert_close(
        sample["target"], sample_data[1, 0:20, :]
    )


def test_getitem_with_station_features(
    sample_data, sample_mask, sample_timestamps, sample_station_features
):
    """Test __getitem__ with station features."""
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        station_features=sample_station_features,
        window_size=20,
        stride=10,
        apply_synthetic_mask=False,
    )

    sample = dataset[0]
    assert "station_features" in sample
    assert sample["station_features"].shape == (3,)
    torch.testing.assert_close(
        sample["station_features"], sample_station_features[0]
    )


def test_getitem_without_station_features(
    sample_data, sample_mask, sample_timestamps
):
    """Test __getitem__ without station features."""
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
        apply_synthetic_mask=False,
    )

    sample = dataset[0]
    assert "station_features" not in sample


def test_getitem_index_validation(sample_data, sample_mask, sample_timestamps):
    """Test __getitem__ index validation."""
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
    )

    # Valid indices
    _ = dataset[0]
    _ = dataset[44]  # Last valid index

    # Invalid indices
    with pytest.raises(IndexError, match="out of range"):
        _ = dataset[-1]

    with pytest.raises(IndexError, match="out of range"):
        _ = dataset[45]


def test_getitem_observed_masking(sample_data, sample_timestamps):
    """Test that observed tensor correctly masks values."""
    # Create mask with some missing values
    mask = torch.ones(sample_data.shape, dtype=torch.bool)
    mask[0, 5:10, 1] = False  # Mask variable 1 at timesteps 5-10 in first sample

    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
        apply_synthetic_mask=False,
    )

    sample = dataset[0]  # First window of first sample (timesteps 0-19)

    # Check that masked positions are set to 0 in observed
    assert torch.all(sample["observed"][5:10, 1] == 0.0)

    # Check that unmasked positions preserve original values
    torch.testing.assert_close(
        sample["observed"][0:5, 1], sample_data[0, 0:5, 1]
    )

    # Check that target always has ground truth
    torch.testing.assert_close(
        sample["target"], sample_data[0, 0:20, :]
    )


# ========================================
# Synthetic Masking Tests
# ========================================


def test_synthetic_masking_mcar(sample_data, sample_mask, sample_timestamps):
    """Test synthetic MCAR masking."""
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
        masking_strategy="mcar",
        masking_config={"missing_ratio": 0.3, "seed": 42},
        apply_synthetic_mask=True,
    )

    sample = dataset[0]

    # Check that mask has some missing values
    assert not torch.all(sample["mask"])

    # Check that observed has zeros at masked positions
    assert torch.all(sample["observed"][~sample["mask"]] == 0.0)


def test_synthetic_masking_disabled(sample_data, sample_mask, sample_timestamps):
    """Test that synthetic masking can be disabled."""
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
        masking_strategy="mcar",
        masking_config={"missing_ratio": 0.3},
        apply_synthetic_mask=False,  # Disabled
    )

    sample = dataset[0]

    # All values should be observed (original mask is all True)
    assert torch.all(sample["mask"])


def test_synthetic_masking_none_strategy(sample_data, sample_mask, sample_timestamps):
    """Test with masking_strategy=None."""
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
        masking_strategy=None,  # No synthetic masking
        apply_synthetic_mask=True,
    )

    sample = dataset[0]

    # All values should be observed
    assert torch.all(sample["mask"])


# ========================================
# Edge Cases
# ========================================


def test_single_window_per_sample(sample_data, sample_mask, sample_timestamps):
    """Test with window_size equal to sequence length."""
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=100,  # Full sequence
        stride=1,
        apply_synthetic_mask=False,
    )

    assert dataset.windows_per_sample == 1
    assert len(dataset) == 5  # Only 1 window per sample

    sample = dataset[0]
    assert sample["observed"].shape == (100, 3)


def test_stride_equals_window_size(sample_data, sample_mask, sample_timestamps):
    """Test with stride equal to window size (no overlap)."""
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=25,
        stride=25,
        apply_synthetic_mask=False,
    )

    # (100 - 25) // 25 + 1 = 4 windows per sample
    assert dataset.windows_per_sample == 4
    assert len(dataset) == 20  # 5 samples * 4 windows

    # Check that windows don't overlap
    sample0 = dataset[0]
    sample1 = dataset[1]
    assert sample0["window_start"].item() == 0
    assert sample1["window_start"].item() == 25


def test_small_stride(sample_data, sample_mask, sample_timestamps):
    """Test with stride=1 (maximum overlap)."""
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=1,
        apply_synthetic_mask=False,
    )

    # (100 - 20) // 1 + 1 = 81 windows per sample
    assert dataset.windows_per_sample == 81
    assert len(dataset) == 405  # 5 samples * 81 windows


# ========================================
# Integration Tests
# ========================================


def test_dataset_iteration(sample_data, sample_mask, sample_timestamps):
    """Test iterating over entire dataset."""
    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
        apply_synthetic_mask=False,
    )

    samples = [dataset[i] for i in range(len(dataset))]
    assert len(samples) == 45

    # Check that all samples have consistent shapes
    for sample in samples:
        assert sample["observed"].shape == (20, 3)
        assert sample["mask"].shape == (20, 3)


def test_dataset_with_dataloader(sample_data, sample_mask, sample_timestamps):
    """Test dataset with PyTorch DataLoader."""
    from torch.utils.data import DataLoader

    dataset = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
        apply_synthetic_mask=False,
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    batch = next(iter(dataloader))

    # Check batch shapes
    assert batch["observed"].shape == (8, 20, 3)  # (B, T, V)
    assert batch["mask"].shape == (8, 20, 3)
    assert batch["target"].shape == (8, 20, 3)
    assert batch["timestamps"].shape == (8, 20)
    assert batch["sample_idx"].shape == (8,)
    assert batch["window_start"].shape == (8,)


def test_dataloader_collation(sample_data, sample_mask, sample_timestamps):
    """Test comprehensive DataLoader collation scenarios."""
    from torch.utils.data import DataLoader

    # Test 1: Basic collation without station features
    dataset_no_features = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
        apply_synthetic_mask=False,
    )

    dataloader_no_features = DataLoader(
        dataset_no_features, batch_size=8, shuffle=False
    )
    batch = next(iter(dataloader_no_features))

    # Verify batch shapes
    assert batch["observed"].shape == (8, 20, 3)
    assert batch["mask"].shape == (8, 20, 3)
    assert batch["target"].shape == (8, 20, 3)
    assert batch["timestamps"].shape == (8, 20)
    assert batch["sample_idx"].shape == (8,)
    assert batch["window_start"].shape == (8,)
    assert "station_features" not in batch

    # Test 2: Collation with station features
    station_features = torch.randn(5, 4)  # 5 stations, 4 features
    dataset_with_features = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        station_features=station_features,
        window_size=20,
        stride=10,
        apply_synthetic_mask=False,
    )

    dataloader_with_features = DataLoader(
        dataset_with_features, batch_size=8, shuffle=False
    )
    batch = next(iter(dataloader_with_features))

    assert batch["station_features"].shape == (8, 4)

    # Test 3: Collation with synthetic masking
    dataset_with_masking = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
        masking_strategy="mcar",
        masking_config={"missing_ratio": 0.2},
        apply_synthetic_mask=True,
    )

    dataloader_with_masking = DataLoader(
        dataset_with_masking, batch_size=8, shuffle=False, num_workers=0
    )
    batch = next(iter(dataloader_with_masking))

    # Verify shapes still correct with synthetic masking
    assert batch["observed"].shape == (8, 20, 3)
    assert batch["mask"].shape == (8, 20, 3)

    # Test 4: Shuffle and different batch sizes
    dataloader_shuffled = DataLoader(
        dataset_no_features, batch_size=3, shuffle=True
    )
    batch = next(iter(dataloader_shuffled))

    assert batch["observed"].shape == (3, 20, 3)

    # Test 5: Drop last batch
    dataloader_drop_last = DataLoader(
        dataset_no_features, batch_size=10, shuffle=False, drop_last=True
    )
    batches = list(dataloader_drop_last)

    # Should have 4 batches (45 samples // 10 = 4 full batches)
    assert len(batches) == 4
    for batch in batches:
        assert batch["observed"].shape == (10, 20, 3)


def test_reproducibility_with_seed(sample_data, sample_mask, sample_timestamps):
    """Test that seeded synthetic masking is reproducible."""
    dataset1 = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
        masking_strategy="mcar",
        masking_config={"missing_ratio": 0.3, "seed": 42},
        apply_synthetic_mask=True,
    )

    dataset2 = TimeSeriesImputationDataset(
        data=sample_data,
        mask=sample_mask,
        timestamps=sample_timestamps,
        window_size=20,
        stride=10,
        masking_strategy="mcar",
        masking_config={"missing_ratio": 0.3, "seed": 42},
        apply_synthetic_mask=True,
    )

    sample1 = dataset1[0]
    sample2 = dataset2[0]

    torch.testing.assert_close(sample1["mask"], sample2["mask"])
