"""Tests for evaluation metrics."""

import math

import pytest
import torch

from weather_imputation.evaluation.metrics import (
    compute_all_metrics,
    compute_bias,
    compute_mae,
    compute_mse,
    compute_r2_score,
    compute_rmse,
)


class TestComputeRMSE:
    """Tests for compute_rmse function."""

    def test_perfect_prediction(self):
        """Test RMSE with perfect predictions."""
        y_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        y_pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        mask = torch.tensor([[[True, True], [True, True]]])

        rmse = compute_rmse(y_true, y_pred, mask)

        assert rmse == 0.0

    def test_known_rmse(self):
        """Test RMSE with known errors."""
        y_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        y_pred = torch.tensor([[[2.0, 2.0], [3.0, 5.0]]])  # errors: [1, 0, 0, 1]
        mask = torch.tensor([[[True, True], [True, True]]])

        rmse = compute_rmse(y_true, y_pred, mask)

        # RMSE = sqrt((1^2 + 0^2 + 0^2 + 1^2) / 4) = sqrt(0.5) ≈ 0.707
        assert rmse == pytest.approx(math.sqrt(0.5), abs=1e-6)

    def test_partial_mask(self):
        """Test RMSE with partial mask (only some positions evaluated)."""
        y_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        y_pred = torch.tensor([[[1.5, 2.0], [3.0, 5.0]]])
        mask = torch.tensor([[[True, False], [False, True]]])

        rmse = compute_rmse(y_true, y_pred, mask)

        # Only positions (0,0,0) and (0,1,1) evaluated
        # Errors: [0.5, 1.0]
        # RMSE = sqrt((0.5^2 + 1.0^2) / 2) = sqrt(0.625) ≈ 0.791
        assert rmse == pytest.approx(math.sqrt(0.625), abs=1e-6)

    def test_multiple_samples(self):
        """Test RMSE with multiple samples in batch."""
        y_true = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],  # sample 1
            [[5.0, 6.0], [7.0, 8.0]],  # sample 2
        ])  # (2, 2, 2)
        y_pred = torch.tensor([
            [[1.5, 2.0], [3.0, 5.0]],
            [[5.0, 6.5], [7.0, 8.0]],
        ])
        mask = torch.ones_like(y_true, dtype=torch.bool)

        rmse = compute_rmse(y_true, y_pred, mask)

        # Errors: [0.5, 0, 0, 1, 0, 0.5, 0, 0]
        # RMSE = sqrt((0.5^2 + 1^2 + 0.5^2) / 8) = sqrt(1.5 / 8) ≈ 0.433
        assert rmse == pytest.approx(math.sqrt(1.5 / 8), abs=1e-6)

    def test_empty_mask(self):
        """Test RMSE with no positions to evaluate (returns NaN)."""
        y_true = torch.tensor([[[1.0, 2.0]]])
        y_pred = torch.tensor([[[1.5, 2.5]]])
        mask = torch.tensor([[[False, False]]])

        rmse = compute_rmse(y_true, y_pred, mask)

        assert math.isnan(rmse)

    def test_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        y_true = torch.tensor([[[1.0, 2.0]]])  # (1, 1, 2)
        y_pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        mask = torch.tensor([[[True, True]]])  # (1, 1, 2)

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_rmse(y_true, y_pred, mask)

    def test_wrong_mask_dtype(self):
        """Test that non-bool mask raises TypeError."""
        y_true = torch.tensor([[[1.0, 2.0]]])
        y_pred = torch.tensor([[[1.5, 2.5]]])
        mask = torch.tensor([[[1, 0]]], dtype=torch.int32)

        with pytest.raises(TypeError, match="mask must be bool dtype"):
            compute_rmse(y_true, y_pred, mask)

    def test_device_consistency(self):
        """Test RMSE works with GPU tensors if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        y_true = torch.tensor([[[1.0, 2.0]]]).cuda()
        y_pred = torch.tensor([[[1.5, 2.5]]]).cuda()
        mask = torch.tensor([[[True, True]]]).cuda()

        rmse = compute_rmse(y_true, y_pred, mask)

        assert isinstance(rmse, float)
        assert rmse == pytest.approx(math.sqrt(0.125), abs=1e-6)


class TestComputeMAE:
    """Tests for compute_mae function."""

    def test_perfect_prediction(self):
        """Test MAE with perfect predictions."""
        y_true = torch.tensor([[[1.0, 2.0]]])
        y_pred = torch.tensor([[[1.0, 2.0]]])
        mask = torch.tensor([[[True, True]]])

        mae = compute_mae(y_true, y_pred, mask)

        assert mae == 0.0

    def test_known_mae(self):
        """Test MAE with known errors."""
        y_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        y_pred = torch.tensor([[[2.0, 2.0], [3.0, 5.0]]])
        mask = torch.tensor([[[True, True], [True, True]]])

        mae = compute_mae(y_true, y_pred, mask)

        # MAE = (|1| + |0| + |0| + |1|) / 4 = 0.5
        assert mae == 0.5

    def test_partial_mask(self):
        """Test MAE with partial mask."""
        y_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        y_pred = torch.tensor([[[1.5, 2.0], [3.0, 5.0]]])
        mask = torch.tensor([[[True, False], [False, True]]])

        mae = compute_mae(y_true, y_pred, mask)

        # Only positions (0,0,0) and (0,1,1) evaluated
        # Errors: [0.5, 1.0]
        # MAE = (0.5 + 1.0) / 2 = 0.75
        assert mae == 0.75

    def test_empty_mask(self):
        """Test MAE with no positions to evaluate."""
        y_true = torch.tensor([[[1.0, 2.0]]])
        y_pred = torch.tensor([[[1.5, 2.5]]])
        mask = torch.tensor([[[False, False]]])

        mae = compute_mae(y_true, y_pred, mask)

        assert math.isnan(mae)


class TestComputeBias:
    """Tests for compute_bias function."""

    def test_no_bias(self):
        """Test bias with unbiased predictions."""
        y_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        y_pred = torch.tensor([[[2.0, 1.0], [3.0, 5.0]]])  # +1, -1, 0, +1
        mask = torch.tensor([[[True, True], [True, True]]])

        bias = compute_bias(y_true, y_pred, mask)

        # Bias = mean([+1, -1, 0, +1]) = 1/4 = 0.25
        assert bias == 0.25

    def test_positive_bias(self):
        """Test positive bias (overestimation)."""
        y_true = torch.tensor([[[1.0, 2.0]]])
        y_pred = torch.tensor([[[2.0, 3.0]]])
        mask = torch.tensor([[[True, True]]])

        bias = compute_bias(y_true, y_pred, mask)

        # Bias = mean([+1, +1]) = +1.0
        assert bias == 1.0

    def test_negative_bias(self):
        """Test negative bias (underestimation)."""
        y_true = torch.tensor([[[3.0, 4.0]]])
        y_pred = torch.tensor([[[2.0, 3.0]]])
        mask = torch.tensor([[[True, True]]])

        bias = compute_bias(y_true, y_pred, mask)

        # Bias = mean([-1, -1]) = -1.0
        assert bias == -1.0

    def test_partial_mask(self):
        """Test bias with partial mask."""
        y_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        y_pred = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])
        mask = torch.tensor([[[True, False], [False, True]]])

        bias = compute_bias(y_true, y_pred, mask)

        # Only positions (0,0,0) and (0,1,1) evaluated
        # Errors: [+1, +1]
        # Bias = mean([+1, +1]) = +1.0
        assert bias == 1.0


class TestComputeR2Score:
    """Tests for compute_r2_score function."""

    def test_perfect_prediction(self):
        """Test R² with perfect predictions."""
        y_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        y_pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        mask = torch.tensor([[[True, True], [True, True]]])

        r2 = compute_r2_score(y_true, y_pred, mask)

        assert r2 == 1.0

    def test_mean_baseline(self):
        """Test R² when predictions equal mean (R²=0)."""
        y_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # mean = 2.5
        y_pred = torch.tensor([[[2.5, 2.5], [2.5, 2.5]]])
        mask = torch.tensor([[[True, True], [True, True]]])

        r2 = compute_r2_score(y_true, y_pred, mask)

        # When predictions = mean, SS_res = SS_tot, so R² = 0
        assert r2 == pytest.approx(0.0, abs=1e-6)

    def test_worse_than_mean(self):
        """Test R² when predictions are worse than mean (R²<0)."""
        y_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # mean = 2.5
        y_pred = torch.tensor([[[0.0, 5.0], [0.0, 5.0]]])  # very bad predictions
        mask = torch.tensor([[[True, True], [True, True]]])

        r2 = compute_r2_score(y_true, y_pred, mask)

        # R² should be negative
        assert r2 < 0.0

    def test_good_prediction(self):
        """Test R² with good (but not perfect) predictions."""
        y_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        y_pred = torch.tensor([[[1.1, 2.1], [2.9, 3.9]]])
        mask = torch.tensor([[[True, True], [True, True]]])

        r2 = compute_r2_score(y_true, y_pred, mask)

        # Should be very high (close to 1.0) due to small errors
        assert 0.8 < r2 < 1.0

    def test_constant_values(self):
        """Test R² with constant ground truth (returns NaN)."""
        y_true = torch.tensor([[[5.0, 5.0], [5.0, 5.0]]])
        y_pred = torch.tensor([[[5.0, 5.0], [5.0, 5.0]]])
        mask = torch.tensor([[[True, True], [True, True]]])

        r2 = compute_r2_score(y_true, y_pred, mask)

        # Zero variance in ground truth -> NaN
        assert math.isnan(r2)

    def test_partial_mask(self):
        """Test R² with partial mask."""
        y_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        y_pred = torch.tensor([[[1.0, 99.0], [3.0, 4.0]]])
        mask = torch.tensor([[[True, False], [True, True]]])

        r2 = compute_r2_score(y_true, y_pred, mask)

        # Only positions (0,0,0), (0,1,0), (0,1,1) evaluated
        # All perfect predictions -> R² = 1.0
        assert r2 == 1.0


class TestComputeMSE:
    """Tests for compute_mse function."""

    def test_perfect_prediction(self):
        """Test MSE with perfect predictions."""
        y_true = torch.tensor([[[1.0, 2.0]]])
        y_pred = torch.tensor([[[1.0, 2.0]]])
        mask = torch.tensor([[[True, True]]])

        mse = compute_mse(y_true, y_pred, mask)

        assert mse == 0.0

    def test_known_mse(self):
        """Test MSE with known errors."""
        y_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        y_pred = torch.tensor([[[2.0, 2.0], [3.0, 5.0]]])
        mask = torch.tensor([[[True, True], [True, True]]])

        mse = compute_mse(y_true, y_pred, mask)

        # MSE = (1^2 + 0^2 + 0^2 + 1^2) / 4 = 0.5
        assert mse == 0.5

    def test_mse_rmse_relationship(self):
        """Test that RMSE = sqrt(MSE)."""
        y_true = torch.randn(2, 10, 3)
        y_pred = y_true + torch.randn(2, 10, 3) * 0.1
        mask = torch.ones_like(y_true, dtype=torch.bool)

        mse = compute_mse(y_true, y_pred, mask)
        rmse = compute_rmse(y_true, y_pred, mask)

        assert rmse == pytest.approx(math.sqrt(mse), abs=1e-6)


class TestComputeAllMetrics:
    """Tests for compute_all_metrics convenience function."""

    def test_returns_all_metrics(self):
        """Test that all metrics are computed and returned."""
        y_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        y_pred = torch.tensor([[[1.5, 2.0], [3.0, 5.0]]])
        mask = torch.tensor([[[True, True], [True, True]]])

        metrics = compute_all_metrics(y_true, y_pred, mask)

        assert set(metrics.keys()) == {"rmse", "mae", "mse", "bias", "r2"}
        assert all(isinstance(v, float) for v in metrics.values())
        assert all(not math.isnan(v) for v in metrics.values())

    def test_perfect_prediction_all_metrics(self):
        """Test all metrics with perfect predictions."""
        y_true = torch.randn(3, 20, 5)
        y_pred = y_true.clone()
        mask = torch.ones_like(y_true, dtype=torch.bool)

        metrics = compute_all_metrics(y_true, y_pred, mask)

        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["bias"] == 0.0
        assert metrics["r2"] == 1.0

    def test_empty_mask_all_metrics(self):
        """Test all metrics with empty mask."""
        y_true = torch.tensor([[[1.0, 2.0]]])
        y_pred = torch.tensor([[[1.5, 2.5]]])
        mask = torch.tensor([[[False, False]]])

        metrics = compute_all_metrics(y_true, y_pred, mask)

        assert all(math.isnan(v) for v in metrics.values())


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_large_batch(self):
        """Test metrics with large batch size."""
        y_true = torch.randn(100, 50, 6)
        y_pred = y_true + torch.randn(100, 50, 6) * 0.5
        mask = torch.ones_like(y_true, dtype=torch.bool)

        rmse = compute_rmse(y_true, y_pred, mask)

        assert isinstance(rmse, float)
        assert not math.isnan(rmse)
        assert rmse > 0

    def test_single_value(self):
        """Test metrics with single value."""
        y_true = torch.tensor([[[5.0]]])
        y_pred = torch.tensor([[[6.0]]])
        mask = torch.tensor([[[True]]])

        rmse = compute_rmse(y_true, y_pred, mask)
        mae = compute_mae(y_true, y_pred, mask)
        bias = compute_bias(y_true, y_pred, mask)

        assert rmse == 1.0
        assert mae == 1.0
        assert bias == 1.0

    def test_mixed_mask_pattern(self):
        """Test metrics with complex mask pattern."""
        y_true = torch.randn(5, 20, 4)
        y_pred = y_true + torch.randn(5, 20, 4) * 0.2

        # Create checkerboard-like mask
        mask = torch.zeros_like(y_true, dtype=torch.bool)
        mask[::2, ::2, ::2] = True
        mask[1::2, 1::2, 1::2] = True

        metrics = compute_all_metrics(y_true, y_pred, mask)

        assert all(isinstance(v, float) for v in metrics.values())
        assert all(not math.isnan(v) for v in metrics.values())

    def test_extreme_values(self):
        """Test metrics with extreme values."""
        y_true = torch.tensor([[[1e6, 1e-6]]])
        y_pred = torch.tensor([[[1e6 + 100, 1e-6 + 1e-7]]])
        mask = torch.tensor([[[True, True]]])

        metrics = compute_all_metrics(y_true, y_pred, mask)

        assert all(isinstance(v, float) for v in metrics.values())
        assert all(not math.isnan(v) for v in metrics.values())
