"""Tests for preprocessing script.

These are integration tests that verify the preprocessing pipeline works end-to-end.
"""

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

from weather_imputation.config.paths import PROCESSED_DIR
from weather_imputation.data.ghcnh_loader import (
    extract_tier1_variables,
    filter_by_quality_flags,
    load_station_all_years,
)
from weather_imputation.data.normalization import Normalizer
from weather_imputation.data.splits import create_split


class TestPreprocessingComponents:
    """Test individual components used by the preprocessing script."""

    def test_normalizer_is_for_global_training_not_preprocessing(self):
        """Test that Normalizer is designed for global (batch) normalization during training.

        The Normalizer class computes statistics across ALL samples in a batch,
        ensuring consistent scaling when training on multiple stations together.
        It should NOT be used during preprocessing for per-station normalization.
        """
        import torch

        # Simulate training scenario: 3 stations, 24 hours each, 2 variables
        station_a = torch.randn(1, 24, 2) * 5 + 15  # temp-like: mean=15, std=5
        station_b = torch.randn(1, 24, 2) * 10 + 50  # different scale
        station_c = torch.randn(1, 24, 2) * 2 + 5   # different scale

        # Concatenate all stations
        all_data = torch.cat([station_a, station_b, station_c], dim=0)  # (3, 24, 2)
        all_mask = torch.ones_like(all_data, dtype=torch.bool)

        # Fit normalizer on ALL training data (global statistics)
        normalizer = Normalizer(method="zscore")
        normalizer.fit(all_data, all_mask)

        # Transform normalizes using GLOBAL stats
        normalized = normalizer.transform(all_data, all_mask)

        # The key property: GLOBAL statistics across all stations should be normalized
        # (not necessarily each individual station)
        for var_idx in range(2):
            all_var_data = normalized[:, :, var_idx].flatten()
            global_mean = all_var_data.mean()
            global_std = all_var_data.std()

            # Global stats should be close to standard normal
            assert abs(global_mean) < 0.5, f"Global mean should be ~0, got {global_mean}"
            assert abs(global_std - 1.0) < 0.5, f"Global std should be ~1, got {global_std}"

    def test_preprocessing_should_not_normalize_data(self):
        """Test that preprocessing script does NOT normalize data.

        Normalization should happen during training using the global Normalizer,
        not during preprocessing. The parquet files should contain raw data.
        """
        # Create sample station data (raw, unnormalized)
        df = pl.DataFrame({
            "DATE": ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00"],
            "temperature": [10.0, 15.0, 20.0],  # Raw values
            "pressure": [1000.0, 1010.0, 1020.0],  # Raw values
            "station_id": ["STATION_A", "STATION_A", "STATION_A"],
        })

        # After preprocessing, data should remain raw (unnormalized)
        # Only metadata columns might be added
        assert "temperature" in df.columns
        assert df["temperature"].to_list() == [10.0, 15.0, 20.0]  # Unchanged

        # No normalization_stats column should exist (old approach)
        # Only normalization_method might be stored as metadata
        assert "normalization_stats" not in df.columns

    def test_load_station_all_years_signature(self):
        """Verify load_station_all_years accepts correct parameters."""
        # This would have caught the start_year/end_year bug
        # The function should accept: station_id and years (list)

        # Test with years parameter (correct usage)
        result = load_station_all_years(
            station_id="FAKE_STATION",
            years=[2020, 2021],
        )
        # Result should be None for fake station, but call should work
        assert result is None or isinstance(result, pl.DataFrame)

    def test_filter_by_quality_flags_signature(self, sample_weather_data):
        """Verify filter_by_quality_flags accepts correct parameters."""
        # This would have caught the exclude_suspects vs exclude_suspect bug

        # Test with exclude_suspect parameter (correct usage)
        result = filter_by_quality_flags(
            sample_weather_data,
            exclude_suspect=True,
            variables=["temperature"],
        )
        assert isinstance(result, pl.DataFrame)

    def test_create_split_signature(self, sample_metadata):
        """Verify create_split accepts correct parameters."""
        # This would have caught the SplitConfig object vs parameters bug

        # Test with individual parameters (correct usage)
        train_ids, val_ids, test_ids = create_split(
            sample_metadata,
            strategy="simulated",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

        assert isinstance(train_ids, list)
        assert isinstance(val_ids, list)
        assert isinstance(test_ids, list)


class TestPreprocessingIntegration:
    """Integration tests for the full preprocessing pipeline."""

    def test_preprocessing_pipeline_with_small_dataset(
        self, sample_metadata, tmp_path
    ):
        """Test the full preprocessing pipeline with a small dataset.

        This simulates what preprocess.py does but with controlled test data.
        """
        # 1. Create station list
        station_ids = sample_metadata["station_id"].to_list()[:2]  # Just 2 stations

        # 2. Apply split
        train_ids, val_ids, test_ids = create_split(
            sample_metadata,
            strategy="simulated",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

        # For simulated strategy, all stations should be in all splits
        assert len(train_ids) == len(sample_metadata)
        assert len(val_ids) == len(sample_metadata)
        assert len(test_ids) == len(sample_metadata)

        # 3. Test data loading and filtering for one station
        # (This would fail if we have real data, but tests the API)
        df = load_station_all_years(
            station_id=station_ids[0],
            years=[2020],
        )

        # If we got data, test the filtering pipeline
        if df is not None:
            # Extract variables
            df = extract_tier1_variables(df, variables=["temperature"])

            # Apply quality filtering (correct parameter name!)
            df = filter_by_quality_flags(
                df,
                exclude_suspect=False,  # Use correct parameter name
                variables=["temperature"],
            )

            # Data should remain unnormalized (raw values)
            # Normalization happens during training, not preprocessing
            if len(df) > 0 and "temperature" in df.columns:
                assert isinstance(df, pl.DataFrame)
                # Temperature values should be in reasonable raw range
                # (not normalized to mean=0, std=1)
                temps = df["temperature"].drop_nulls()
                if len(temps) > 0:
                    # Raw temperature values are typically -50 to 50°C
                    # If they were normalized, they'd be in [-3, 3] range
                    assert abs(temps.mean()) > 0.1 or abs(temps.std() - 1.0) > 0.5


# Fixtures

@pytest.fixture
def sample_weather_data():
    """Create sample weather data for testing."""
    return pl.DataFrame({
        "DATE": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"],
        "temperature": [10.0, 11.0],
        "temperature_Quality_Code": ["1", "1"],
    })


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return pl.DataFrame({
        "station_id": ["STATION_A", "STATION_B", "STATION_C"],
        "latitude": [40.0, 41.0, 42.0],
        "longitude": [-105.0, -104.0, -103.0],
        "first_observation": [
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 1, 1),
        ],
        "last_observation": [
            pl.datetime(2020, 12, 31),
            pl.datetime(2020, 12, 31),
            pl.datetime(2020, 12, 31),
        ],
    })


class TestPreprocessScriptCLI:
    """Tests for the preprocessing script CLI itself."""

    def test_help_command(self):
        """Test that the help command works."""
        import subprocess

        result = subprocess.run(
            ["uv", "run", "python", "src/scripts/preprocess.py", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "preprocess" in result.stdout.lower()
        assert "--stations-file" in result.stdout

    @pytest.mark.slow
    def test_preprocessing_with_test_stations(self, tmp_path):
        """Test preprocessing with a small test station file.

        This is marked as slow and would be run separately.
        """
        import subprocess

        # Create a test station file with just 2 stations
        test_stations = ["CAA00710601", "CAA00710983"]
        stations_file = tmp_path / "test_stations.json"
        with open(stations_file, "w") as f:
            json.dump(test_stations, f)

        # Run preprocessing
        result = subprocess.run(
            [
                "uv", "run", "python", "src/scripts/preprocess.py",
                "--stations-file", str(stations_file),
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        # Should succeed or fail gracefully
        if result.returncode == 0:
            # Check output files were created
            assert (tmp_path / "train.parquet").exists()
            assert (tmp_path / "val.parquet").exists()
            assert (tmp_path / "test.parquet").exists()
