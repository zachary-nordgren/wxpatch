"""Tests for classical imputation models."""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from weather_imputation.models.base import Imputer
from weather_imputation.models.classical.linear import LinearInterpolationImputer

# ==============================================================================
# Test LinearInterpolationImputer
# ==============================================================================


class TestLinearInterpolation:
    """Tests for LinearInterpolationImputer."""

    def test_initialization(self):
        """Test basic initialization."""
        imputer = LinearInterpolationImputer()
        assert imputer.name == "Linear Interpolation"
        assert not imputer.is_fitted
        assert imputer.max_gap_length is None

    def test_initialization_with_max_gap_length(self):
        """Test initialization with max_gap_length parameter."""
        imputer = LinearInterpolationImputer(max_gap_length=24)
        assert imputer.max_gap_length == 24
        assert imputer.get_hyperparameters() == {"max_gap_length": 24}

    def test_protocol_compliance(self):
        """Test that LinearInterpolationImputer implements Imputer protocol."""
        imputer = LinearInterpolationImputer()
        assert isinstance(imputer, Imputer)

    def test_fit_marks_as_fitted(self):
        """Test that fit() marks the imputer as fitted."""
        imputer = LinearInterpolationImputer()
        assert not imputer.is_fitted

        # Create dummy DataLoader
        dummy_data = torch.randn(4, 10, 3)
        dummy_mask = torch.ones(4, 10, 3, dtype=torch.bool)
        dummy_loader = DataLoader(TensorDataset(dummy_data, dummy_mask), batch_size=2)

        imputer.fit(dummy_loader)
        assert imputer.is_fitted

    def test_impute_before_fit_raises(self):
        """Test that impute() raises if called before fit()."""
        imputer = LinearInterpolationImputer()
        observed = torch.randn(2, 10, 3)
        mask = torch.ones(2, 10, 3, dtype=torch.bool)

        with pytest.raises(RuntimeError, match="has not been fitted"):
            imputer.impute(observed, mask)

    def test_impute_preserves_observed_values(self):
        """Test that impute() preserves observed values."""
        imputer = LinearInterpolationImputer()
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 5, 1)), batch_size=1))

        # Create data: [1, ?, 3, ?, 5]
        observed = torch.tensor([[[1.0], [999.0], [3.0], [999.0], [5.0]]])
        mask = torch.tensor([[[True], [False], [True], [False], [True]]])

        result = imputer.impute(observed, mask)

        # Check that observed values are preserved
        assert result[0, 0, 0] == 1.0
        assert result[0, 2, 0] == 3.0
        assert result[0, 4, 0] == 5.0

    def test_impute_simple_gap(self):
        """Test linear interpolation fills a simple gap correctly."""
        imputer = LinearInterpolationImputer()
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 5, 1)), batch_size=1))

        # Create data: [1, ?, 3, ?, 5]
        observed = torch.tensor([[[1.0], [0.0], [3.0], [0.0], [5.0]]])
        mask = torch.tensor([[[True], [False], [True], [False], [True]]])

        result = imputer.impute(observed, mask)

        # Check interpolated values
        # Between 1 and 3: should be 2
        assert torch.isclose(result[0, 1, 0], torch.tensor(2.0))
        # Between 3 and 5: should be 4
        assert torch.isclose(result[0, 3, 0], torch.tensor(4.0))

    def test_impute_multiple_gaps(self):
        """Test interpolation with multiple gaps."""
        imputer = LinearInterpolationImputer()
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 7, 1)), batch_size=1))

        # Create data: [0, ?, 2, ?, ?, 5, ?]
        observed = torch.tensor([[[0.0], [0.0], [2.0], [0.0], [0.0], [5.0], [0.0]]])
        mask = torch.tensor(
            [[[True], [False], [True], [False], [False], [True], [False]]]
        )

        result = imputer.impute(observed, mask)

        # Gap 1: between 0 and 2 (should be 1)
        assert torch.isclose(result[0, 1, 0], torch.tensor(1.0))

        # Gap 2: between 2 and 5 (should be 3, 4)
        assert torch.isclose(result[0, 3, 0], torch.tensor(3.0))
        assert torch.isclose(result[0, 4, 0], torch.tensor(4.0))

    def test_impute_leading_missing_values(self):
        """Test that leading missing values are forward-filled."""
        imputer = LinearInterpolationImputer()
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 5, 1)), batch_size=1))

        # Create data: [?, ?, 3, ?, 5]
        observed = torch.tensor([[[0.0], [0.0], [3.0], [0.0], [5.0]]])
        mask = torch.tensor([[[False], [False], [True], [False], [True]]])

        result = imputer.impute(observed, mask)

        # Leading missing values should be forward-filled from first observed (3)
        assert result[0, 0, 0] == 3.0
        assert result[0, 1, 0] == 3.0

    def test_impute_trailing_missing_values(self):
        """Test that trailing missing values are backward-filled."""
        imputer = LinearInterpolationImputer()
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 5, 1)), batch_size=1))

        # Create data: [1, ?, 3, ?, ?]
        observed = torch.tensor([[[1.0], [0.0], [3.0], [0.0], [0.0]]])
        mask = torch.tensor([[[True], [False], [True], [False], [False]]])

        result = imputer.impute(observed, mask)

        # Trailing missing values should be backward-filled from last observed (3)
        assert result[0, 3, 0] == 3.0
        assert result[0, 4, 0] == 3.0

    def test_impute_all_observed(self):
        """Test that no interpolation occurs when all values are observed."""
        imputer = LinearInterpolationImputer()
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 5, 1)), batch_size=1))

        observed = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
        mask = torch.ones(1, 5, 1, dtype=torch.bool)

        result = imputer.impute(observed, mask)

        # All values should be unchanged
        assert torch.equal(result, observed)

    def test_impute_all_missing(self):
        """Test that all-missing series returns zeros."""
        imputer = LinearInterpolationImputer()
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 5, 1)), batch_size=1))

        observed = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
        mask = torch.zeros(1, 5, 1, dtype=torch.bool)

        result = imputer.impute(observed, mask)

        # All values should be zero
        assert torch.equal(result, torch.zeros(1, 5, 1))

    def test_impute_multiple_variables(self):
        """Test interpolation with multiple variables."""
        imputer = LinearInterpolationImputer()
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 5, 2)), batch_size=1))

        # Variable 0: [1, ?, 3, ?, 5]
        # Variable 1: [10, ?, ?, 40, 50]
        observed = torch.tensor([[[1.0, 10.0], [0.0, 0.0], [3.0, 0.0], [0.0, 40.0], [5.0, 50.0]]])
        mask = torch.tensor(
            [
                [
                    [True, True],
                    [False, False],
                    [True, False],
                    [False, True],
                    [True, True],
                ]
            ]
        )

        result = imputer.impute(observed, mask)

        # Variable 0: check interpolation
        assert torch.isclose(result[0, 1, 0], torch.tensor(2.0))
        assert torch.isclose(result[0, 3, 0], torch.tensor(4.0))

        # Variable 1: check interpolation
        assert torch.isclose(result[0, 1, 1], torch.tensor(20.0))
        assert torch.isclose(result[0, 2, 1], torch.tensor(30.0))

    def test_impute_multiple_samples(self):
        """Test interpolation with multiple samples in batch."""
        imputer = LinearInterpolationImputer()
        imputer.fit(DataLoader(TensorDataset(torch.randn(2, 5, 1)), batch_size=1))

        # Sample 0: [1, ?, 3, ?, 5]
        # Sample 1: [0, ?, ?, ?, 4]
        observed = torch.tensor(
            [
                [[1.0], [0.0], [3.0], [0.0], [5.0]],
                [[0.0], [0.0], [0.0], [0.0], [4.0]],
            ]
        )
        mask = torch.tensor(
            [
                [[True], [False], [True], [False], [True]],
                [[True], [False], [False], [False], [True]],
            ]
        )

        result = imputer.impute(observed, mask)

        # Sample 0: check interpolation
        assert torch.isclose(result[0, 1, 0], torch.tensor(2.0))

        # Sample 1: check interpolation (0 to 4 over 4 steps)
        assert torch.isclose(result[1, 1, 0], torch.tensor(1.0))
        assert torch.isclose(result[1, 2, 0], torch.tensor(2.0))
        assert torch.isclose(result[1, 3, 0], torch.tensor(3.0))

    def test_max_gap_length_respected(self):
        """Test that gaps longer than max_gap_length are not filled."""
        imputer = LinearInterpolationImputer(max_gap_length=2)
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 7, 1)), batch_size=1))

        # Create data: [1, ?, 3, ?, ?, ?, 7]
        # Gap 1: length 1 (should be filled)
        # Gap 2: length 3 (should NOT be filled)
        observed = torch.tensor([[[1.0], [0.0], [3.0], [0.0], [0.0], [0.0], [7.0]]])
        mask = torch.tensor(
            [[[True], [False], [True], [False], [False], [False], [True]]]
        )

        result = imputer.impute(observed, mask)

        # Gap 1 should be filled
        assert torch.isclose(result[0, 1, 0], torch.tensor(2.0))

        # Gap 2 should NOT be filled (remain as observed values = 0)
        assert result[0, 3, 0] == 0.0
        assert result[0, 4, 0] == 0.0
        assert result[0, 5, 0] == 0.0

    def test_max_gap_length_exact_boundary(self):
        """Test that gap exactly at max_gap_length is filled."""
        imputer = LinearInterpolationImputer(max_gap_length=2)
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 5, 1)), batch_size=1))

        # Create data: [1, ?, ?, 4, ?]
        # Gap: length 2 (exactly at max_gap_length, should be filled)
        observed = torch.tensor([[[1.0], [0.0], [0.0], [4.0], [0.0]]])
        mask = torch.tensor([[[True], [False], [False], [True], [False]]])

        result = imputer.impute(observed, mask)

        # Gap should be filled
        assert torch.isclose(result[0, 1, 0], torch.tensor(2.0))
        assert torch.isclose(result[0, 2, 0], torch.tensor(3.0))

    def test_save_and_load(self):
        """Test saving and loading imputer state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "linear_imputer"

            # Create and fit imputer
            imputer = LinearInterpolationImputer(max_gap_length=24)
            imputer.fit(DataLoader(TensorDataset(torch.randn(1, 5, 1)), batch_size=1))

            # Save
            imputer.save(save_path)

            # Check that files were created
            assert (save_path / "metadata.pt").exists()
            assert (save_path / "config.json").exists()

            # Create new imputer and load
            new_imputer = LinearInterpolationImputer()
            assert not new_imputer.is_fitted
            assert new_imputer.max_gap_length is None

            new_imputer.load(save_path)

            # Check that state was restored
            assert new_imputer.is_fitted
            assert new_imputer.max_gap_length == 24
            assert new_imputer.get_hyperparameters() == {"max_gap_length": 24}

    def test_save_load_roundtrip_imputation(self):
        """Test that saved/loaded imputer produces same results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "linear_imputer"

            # Create test data
            observed = torch.tensor([[[1.0], [0.0], [3.0], [0.0], [5.0]]])
            mask = torch.tensor([[[True], [False], [True], [False], [True]]])

            # Original imputer
            imputer1 = LinearInterpolationImputer(max_gap_length=10)
            imputer1.fit(DataLoader(TensorDataset(torch.randn(1, 5, 1)), batch_size=1))
            result1 = imputer1.impute(observed, mask)

            # Save and load
            imputer1.save(save_path)
            imputer2 = LinearInterpolationImputer()
            imputer2.load(save_path)
            result2 = imputer2.impute(observed, mask)

            # Results should be identical
            assert torch.equal(result1, result2)

    def test_input_validation_wrong_type(self):
        """Test that impute() validates input types."""
        imputer = LinearInterpolationImputer()
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 5, 1)), batch_size=1))

        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            imputer.impute([1, 2, 3], torch.ones(1, 3, 1, dtype=torch.bool))

    def test_input_validation_wrong_dimensions(self):
        """Test that impute() validates tensor dimensions."""
        imputer = LinearInterpolationImputer()
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 5, 1)), batch_size=1))

        # 2D tensor (should be 3D)
        with pytest.raises(ValueError, match="must be 3D"):
            imputer.impute(torch.randn(5, 3), torch.ones(5, 3, dtype=torch.bool))

    def test_input_validation_shape_mismatch(self):
        """Test that impute() validates shape consistency."""
        imputer = LinearInterpolationImputer()
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 5, 1)), batch_size=1))

        with pytest.raises(ValueError, match="must have same shape"):
            imputer.impute(torch.randn(1, 5, 3), torch.ones(1, 6, 3, dtype=torch.bool))

    def test_input_validation_mask_dtype(self):
        """Test that impute() validates mask dtype."""
        imputer = LinearInterpolationImputer()
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 5, 1)), batch_size=1))

        with pytest.raises(ValueError, match="must be boolean tensor"):
            imputer.impute(torch.randn(1, 5, 3), torch.ones(1, 5, 3, dtype=torch.float32))

    def test_edge_case_single_timestep(self):
        """Test interpolation with single timestep."""
        imputer = LinearInterpolationImputer()
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 1, 1)), batch_size=1))

        observed = torch.tensor([[[5.0]]])
        mask = torch.tensor([[[True]]])

        result = imputer.impute(observed, mask)
        assert result[0, 0, 0] == 5.0

    def test_edge_case_single_variable(self):
        """Test interpolation with single variable."""
        imputer = LinearInterpolationImputer()
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 5, 1)), batch_size=1))

        observed = torch.tensor([[[1.0], [0.0], [3.0], [0.0], [5.0]]])
        mask = torch.tensor([[[True], [False], [True], [False], [True]]])

        result = imputer.impute(observed, mask)

        # Should interpolate correctly
        assert torch.isclose(result[0, 1, 0], torch.tensor(2.0))
        assert torch.isclose(result[0, 3, 0], torch.tensor(4.0))

    def test_reproducibility(self):
        """Test that imputation is deterministic."""
        imputer = LinearInterpolationImputer()
        imputer.fit(DataLoader(TensorDataset(torch.randn(1, 5, 1)), batch_size=1))

        observed = torch.tensor([[[1.0], [0.0], [3.0], [0.0], [5.0]]])
        mask = torch.tensor([[[True], [False], [True], [False], [True]]])

        result1 = imputer.impute(observed, mask)
        result2 = imputer.impute(observed, mask)

        assert torch.equal(result1, result2)
