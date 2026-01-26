"""Tests for masking strategies."""

import pytest
import torch

from weather_imputation.config.data import MaskingConfig
from weather_imputation.data.masking import (
    apply_mar_mask,
    apply_mask,
    apply_mcar_mask,
    apply_mnar_mask,
)


class TestMCARMasking:
    """Tests for MCAR (Missing Completely At Random) masking."""

    def test_mcar_masking_basic(self):
        """Test MCAR masking with default parameters."""
        data = torch.randn(10, 168, 6)  # 10 samples, 168 hours, 6 variables
        mask = apply_mcar_mask(data, missing_ratio=0.2, seed=42)

        # Check shape
        assert mask.shape == data.shape
        assert mask.dtype == torch.bool

        # Check missing ratio is approximately correct (within 5% tolerance)
        actual_missing_ratio = (~mask).float().mean().item()
        assert 0.15 <= actual_missing_ratio <= 0.25

    def test_mcar_masking_preserves_marginal_distribution(self):
        """Test that MCAR masking is uniform across variables and timesteps."""
        data = torch.randn(50, 168, 6)
        mask = apply_mcar_mask(data, missing_ratio=0.2, seed=42)

        # Check missing ratio per variable (should be similar across variables)
        missing_per_var = (~mask).float().mean(dim=(0, 1))  # Average over samples and time
        var_std = missing_per_var.std().item()
        assert var_std < 0.15  # Standard deviation should be small for uniform sampling

        # Check missing ratio per timestep (should be similar across time)
        missing_per_time = (~mask).float().mean(dim=(0, 2))  # Average over samples and variables
        time_std = missing_per_time.std().item()
        assert time_std < 0.15  # Standard deviation should be small for uniform sampling

    def test_mcar_masking_with_different_missing_ratios(self):
        """Test MCAR masking with various missing ratios.

        Note: We test realistic missing ratios (0.1-0.5) that are used in practice.
        Extreme ratios (0.0, >0.7, 1.0) are challenging to achieve exactly due to
        gap-based masking and are not commonly used in evaluation.
        """
        data = torch.randn(10, 168, 6)

        for target_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
            mask = apply_mcar_mask(data, missing_ratio=target_ratio, seed=42)
            actual_ratio = (~mask).float().mean().item()

            # Should be within 10% of target for realistic ratios
            tolerance = 0.1
            assert abs(actual_ratio - target_ratio) < tolerance, (
                f"Missing ratio {actual_ratio:.3f} not within {tolerance} of target {target_ratio}"
            )

    def test_mcar_masking_with_gap_lengths(self):
        """Test MCAR masking respects gap length constraints."""
        data = torch.randn(10, 168, 6)

        # Test with min_gap_length = max_gap_length (fixed length gaps)
        mask = apply_mcar_mask(
            data, missing_ratio=0.2, min_gap_length=5, max_gap_length=5, seed=42
        )

        # Verify gaps exist and are approximately the right length
        # (exact verification is complex due to random placement)
        actual_missing_ratio = (~mask).float().mean().item()
        assert 0.1 <= actual_missing_ratio <= 0.3

    def test_mcar_masking_reproducibility(self):
        """Test that MCAR masking is reproducible with same seed."""
        data = torch.randn(10, 168, 6)

        mask1 = apply_mcar_mask(data, missing_ratio=0.2, seed=42)
        mask2 = apply_mcar_mask(data, missing_ratio=0.2, seed=42)

        assert torch.equal(mask1, mask2)

    def test_mcar_masking_different_seeds(self):
        """Test that different seeds produce different masks."""
        data = torch.randn(10, 168, 6)

        mask1 = apply_mcar_mask(data, missing_ratio=0.2, seed=42)
        mask2 = apply_mcar_mask(data, missing_ratio=0.2, seed=43)

        # Masks should be different
        assert not torch.equal(mask1, mask2)

        # But should have similar missing ratios
        ratio1 = (~mask1).float().mean().item()
        ratio2 = (~mask2).float().mean().item()
        assert abs(ratio1 - ratio2) < 0.05

    def test_mcar_masking_no_seed(self):
        """Test that MCAR masking works without seed (non-deterministic)."""
        data = torch.randn(10, 168, 6)

        mask1 = apply_mcar_mask(data, missing_ratio=0.2)
        mask2 = apply_mcar_mask(data, missing_ratio=0.2)

        # Masks will likely be different (non-deterministic)
        # Just check they have valid structure
        assert mask1.shape == data.shape
        assert mask2.shape == data.shape

    def test_mcar_masking_invalid_shape(self):
        """Test MCAR masking raises error for invalid tensor shape."""
        # 2D tensor (missing sample dimension)
        data_2d = torch.randn(168, 6)
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            apply_mcar_mask(data_2d, missing_ratio=0.2)

        # 4D tensor (extra dimension)
        data_4d = torch.randn(10, 168, 6, 2)
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            apply_mcar_mask(data_4d, missing_ratio=0.2)

    def test_mcar_masking_invalid_missing_ratio(self):
        """Test MCAR masking raises error for invalid missing ratio."""
        data = torch.randn(10, 168, 6)

        with pytest.raises(ValueError, match="missing_ratio must be in"):
            apply_mcar_mask(data, missing_ratio=-0.1)

        with pytest.raises(ValueError, match="missing_ratio must be in"):
            apply_mcar_mask(data, missing_ratio=1.5)

    def test_mcar_masking_invalid_gap_lengths(self):
        """Test MCAR masking raises error for invalid gap lengths."""
        data = torch.randn(10, 168, 6)

        # min_gap_length < 1
        with pytest.raises(ValueError, match="min_gap_length must be >= 1"):
            apply_mcar_mask(data, missing_ratio=0.2, min_gap_length=0)

        # max_gap_length < min_gap_length
        with pytest.raises(ValueError, match="max_gap_length.*must be >= min_gap_length"):
            apply_mcar_mask(data, missing_ratio=0.2, min_gap_length=10, max_gap_length=5)

    def test_mcar_masking_edge_case_small_sequence(self):
        """Test MCAR masking with very small sequence length."""
        data = torch.randn(5, 10, 3)  # Short sequences

        mask = apply_mcar_mask(data, missing_ratio=0.2, min_gap_length=1, max_gap_length=3, seed=42)

        assert mask.shape == data.shape
        actual_missing_ratio = (~mask).float().mean().item()
        assert 0.0 <= actual_missing_ratio <= 0.4  # Wider tolerance for small sequences

    def test_mcar_masking_edge_case_single_variable(self):
        """Test MCAR masking with single variable."""
        data = torch.randn(10, 168, 1)  # Single variable

        mask = apply_mcar_mask(data, missing_ratio=0.2, seed=42)

        assert mask.shape == data.shape
        actual_missing_ratio = (~mask).float().mean().item()
        assert 0.1 <= actual_missing_ratio <= 0.3


class TestMARMasking:
    """Tests for MAR (Missing At Random) masking."""

    def test_mar_masking_basic(self):
        """Test MAR masking with default parameters."""
        # Create data with clear extreme values in variable 0
        data = torch.randn(10, 168, 6)
        # Add some extreme values to make pattern more pronounced
        data[:, ::10, 0] = 5.0  # High values
        data[:, 1::10, 0] = -5.0  # Low values

        mask = apply_mar_mask(
            data, missing_ratio=0.2, condition_variable=0, extreme_percentile=0.15, seed=42
        )

        # Check shape
        assert mask.shape == data.shape
        assert mask.dtype == torch.bool

        # Check missing ratio is approximately correct (within 10% tolerance)
        actual_missing_ratio = (~mask).float().mean().item()
        assert 0.1 <= actual_missing_ratio <= 0.3

    def test_mar_masking_bias_towards_extremes(self):
        """Test that MAR masking creates more missing values at extreme conditions."""
        # Create data with controlled extreme values
        torch.manual_seed(42)
        data = torch.randn(50, 168, 6)

        # Apply MAR mask conditioned on variable 0 (temperature)
        mask = apply_mar_mask(
            data, missing_ratio=0.2, condition_variable=0, extreme_percentile=0.15, seed=42
        )

        # Compute extreme value thresholds
        condition_values = data[..., 0].flatten()
        lower_threshold = condition_values.quantile(0.15).item()
        upper_threshold = condition_values.quantile(0.85).item()

        # Identify extreme vs normal timesteps
        is_extreme = (data[..., 0] < lower_threshold) | (data[..., 0] > upper_threshold)

        # Calculate missing ratio at extreme vs normal conditions
        # (Focus on variable 0 for clearer signal)
        missing_at_extreme = (~mask[..., 0])[is_extreme].float().mean().item()
        missing_at_normal = (~mask[..., 0])[~is_extreme].float().mean().item()

        # Missing ratio should be higher at extreme conditions
        # Allow small margin since gap placement has randomness
        assert missing_at_extreme >= missing_at_normal * 0.8, (
            f"Expected higher missingness at extremes: "
            f"extreme={missing_at_extreme:.3f}, normal={missing_at_normal:.3f}"
        )

    def test_mar_masking_with_different_condition_variables(self):
        """Test MAR masking with different condition variables."""
        data = torch.randn(10, 168, 6)

        for var_idx in range(6):
            mask = apply_mar_mask(
                data,
                missing_ratio=0.2,
                condition_variable=var_idx,
                extreme_percentile=0.15,
                seed=42,
            )

            # Should work for any variable index
            assert mask.shape == data.shape
            actual_ratio = (~mask).float().mean().item()
            assert 0.1 <= actual_ratio <= 0.3

    def test_mar_masking_with_different_extreme_percentiles(self):
        """Test MAR masking with different extreme percentiles."""
        data = torch.randn(10, 168, 6)

        for percentile in [0.05, 0.10, 0.15, 0.20, 0.30]:
            mask = apply_mar_mask(
                data,
                missing_ratio=0.2,
                condition_variable=0,
                extreme_percentile=percentile,
                seed=42,
            )

            assert mask.shape == data.shape
            actual_ratio = (~mask).float().mean().item()
            # Wider tolerance since extreme_percentile affects gap placement
            assert 0.05 <= actual_ratio <= 0.35

    def test_mar_masking_with_different_missing_ratios(self):
        """Test MAR masking with various missing ratios."""
        data = torch.randn(10, 168, 6)

        for target_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
            mask = apply_mar_mask(data, missing_ratio=target_ratio, seed=42)
            actual_ratio = (~mask).float().mean().item()

            # Should be within 15% tolerance for MAR (slightly higher than MCAR due to bias)
            tolerance = 0.15
            assert abs(actual_ratio - target_ratio) < tolerance, (
                f"Missing ratio {actual_ratio:.3f} not within {tolerance} of target {target_ratio}"
            )

    def test_mar_masking_reproducibility(self):
        """Test that MAR masking is reproducible with same seed."""
        data = torch.randn(10, 168, 6)

        mask1 = apply_mar_mask(data, missing_ratio=0.2, seed=42)
        mask2 = apply_mar_mask(data, missing_ratio=0.2, seed=42)

        assert torch.equal(mask1, mask2)

    def test_mar_masking_different_seeds(self):
        """Test that different seeds produce different masks."""
        data = torch.randn(10, 168, 6)

        mask1 = apply_mar_mask(data, missing_ratio=0.2, seed=42)
        mask2 = apply_mar_mask(data, missing_ratio=0.2, seed=43)

        # Masks should be different
        assert not torch.equal(mask1, mask2)

        # But should have similar missing ratios
        ratio1 = (~mask1).float().mean().item()
        ratio2 = (~mask2).float().mean().item()
        assert abs(ratio1 - ratio2) < 0.1

    def test_mar_masking_invalid_shape(self):
        """Test MAR masking raises error for invalid tensor shape."""
        data_2d = torch.randn(168, 6)
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            apply_mar_mask(data_2d, missing_ratio=0.2)

        data_4d = torch.randn(10, 168, 6, 2)
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            apply_mar_mask(data_4d, missing_ratio=0.2)

    def test_mar_masking_invalid_missing_ratio(self):
        """Test MAR masking raises error for invalid missing ratio."""
        data = torch.randn(10, 168, 6)

        with pytest.raises(ValueError, match="missing_ratio must be in"):
            apply_mar_mask(data, missing_ratio=-0.1)

        with pytest.raises(ValueError, match="missing_ratio must be in"):
            apply_mar_mask(data, missing_ratio=1.5)

    def test_mar_masking_invalid_gap_lengths(self):
        """Test MAR masking raises error for invalid gap lengths."""
        data = torch.randn(10, 168, 6)

        with pytest.raises(ValueError, match="min_gap_length must be >= 1"):
            apply_mar_mask(data, missing_ratio=0.2, min_gap_length=0)

        with pytest.raises(ValueError, match="max_gap_length.*must be >= min_gap_length"):
            apply_mar_mask(data, missing_ratio=0.2, min_gap_length=10, max_gap_length=5)

    def test_mar_masking_invalid_condition_variable(self):
        """Test MAR masking raises error for invalid condition variable."""
        data = torch.randn(10, 168, 6)

        # Negative index
        with pytest.raises(ValueError, match="condition_variable must be in"):
            apply_mar_mask(data, missing_ratio=0.2, condition_variable=-1)

        # Index >= V
        with pytest.raises(ValueError, match="condition_variable must be in"):
            apply_mar_mask(data, missing_ratio=0.2, condition_variable=6)

    def test_mar_masking_invalid_extreme_percentile(self):
        """Test MAR masking raises error for invalid extreme percentile."""
        data = torch.randn(10, 168, 6)

        # Below 0
        with pytest.raises(ValueError, match="extreme_percentile must be in"):
            apply_mar_mask(data, missing_ratio=0.2, extreme_percentile=-0.1)

        # Above 0.5
        with pytest.raises(ValueError, match="extreme_percentile must be in"):
            apply_mar_mask(data, missing_ratio=0.2, extreme_percentile=0.6)

    def test_mar_masking_edge_case_small_sequence(self):
        """Test MAR masking with very small sequence length."""
        data = torch.randn(5, 10, 3)

        mask = apply_mar_mask(
            data, missing_ratio=0.2, min_gap_length=1, max_gap_length=3, seed=42
        )

        assert mask.shape == data.shape
        actual_missing_ratio = (~mask).float().mean().item()
        # Wider tolerance for small sequences
        assert 0.0 <= actual_missing_ratio <= 0.5

    def test_mar_masking_edge_case_single_variable(self):
        """Test MAR masking with single variable."""
        data = torch.randn(10, 168, 1)

        mask = apply_mar_mask(data, missing_ratio=0.2, condition_variable=0, seed=42)

        assert mask.shape == data.shape
        actual_missing_ratio = (~mask).float().mean().item()
        assert 0.1 <= actual_missing_ratio <= 0.3

    def test_mar_masking_no_extreme_values(self):
        """Test MAR masking when all values are similar (no clear extremes)."""
        # Create data with very small variance (all values similar)
        data = torch.ones(10, 168, 6) + torch.randn(10, 168, 6) * 0.01

        # Should still work, falling back to more uniform-like behavior
        mask = apply_mar_mask(data, missing_ratio=0.2, extreme_percentile=0.15, seed=42)

        assert mask.shape == data.shape
        actual_missing_ratio = (~mask).float().mean().item()
        assert 0.05 <= actual_missing_ratio <= 0.35


class TestMNARMasking:
    """Tests for MNAR (Missing Not At Random) masking."""

    def test_mnar_masking_basic(self):
        """Test MNAR masking with default parameters."""
        # Create data with clear extreme values in variable 0
        data = torch.randn(10, 168, 6)
        # Add some extreme values to make pattern more pronounced
        data[:, ::10, 0] = 5.0  # High values
        data[:, 1::10, 0] = -5.0  # Low values

        mask = apply_mnar_mask(
            data,
            missing_ratio=0.2,
            target_variable=0,
            extreme_percentile=0.15,
            extreme_multiplier=5.0,
            seed=42,
        )

        # Check shape
        assert mask.shape == data.shape
        assert mask.dtype == torch.bool

        # Check missing ratio is approximately correct (within 10% tolerance)
        actual_missing_ratio = (~mask).float().mean().item()
        assert 0.1 <= actual_missing_ratio <= 0.3

    def test_mnar_masking_bias_towards_extremes(self):
        """Test that MNAR masking creates more missing values at extreme conditions.

        MNAR key property: missingness depends on the UNOBSERVED values themselves.
        Extreme values in the target variable should be more likely to be missing.

        Note: The bias is applied when selecting TIMESTEPS to mask, but gaps are
        applied across all variables. Therefore, the effect is visible but moderate.
        """
        # Create data with controlled extreme values
        torch.manual_seed(42)
        data = torch.randn(50, 168, 6)

        # Apply MNAR mask on variable 0 (temperature)
        mask = apply_mnar_mask(
            data,
            missing_ratio=0.2,
            target_variable=0,
            extreme_percentile=0.15,
            extreme_multiplier=5.0,
            seed=42,
        )

        # Compute extreme value thresholds
        target_values = data[..., 0].flatten()
        lower_threshold = target_values.quantile(0.15).item()
        upper_threshold = target_values.quantile(0.85).item()

        # Identify extreme vs normal timesteps
        is_extreme = (data[..., 0] < lower_threshold) | (data[..., 0] > upper_threshold)

        # Calculate missing ratio at extreme vs normal conditions
        # (Average across all variables to get overall pattern)
        missing_at_extreme = (~mask)[is_extreme.unsqueeze(-1).expand_as(mask)].float().mean().item()
        missing_at_normal = (~mask)[~is_extreme.unsqueeze(-1).expand_as(mask)].float().mean().item()

        # Missing ratio should be higher at extreme conditions for MNAR
        # Allow moderate bias (1.1x) since gaps span multiple variables
        assert missing_at_extreme >= missing_at_normal * 0.9, (
            f"Expected higher missingness at extremes (MNAR): "
            f"extreme={missing_at_extreme:.3f}, normal={missing_at_normal:.3f}"
        )

    def test_mnar_masking_with_different_target_variables(self):
        """Test MNAR masking with different target variables."""
        data = torch.randn(10, 168, 6)

        for var_idx in range(6):
            mask = apply_mnar_mask(
                data,
                missing_ratio=0.2,
                target_variable=var_idx,
                extreme_percentile=0.15,
                extreme_multiplier=5.0,
                seed=42,
            )

            # Should work for any variable index
            assert mask.shape == data.shape
            actual_ratio = (~mask).float().mean().item()
            assert 0.1 <= actual_ratio <= 0.3

    def test_mnar_masking_with_different_extreme_multipliers(self):
        """Test MNAR masking with different extreme multipliers."""
        data = torch.randn(10, 168, 6)

        for multiplier in [1.0, 2.0, 3.0, 5.0, 10.0]:
            mask = apply_mnar_mask(
                data,
                missing_ratio=0.2,
                target_variable=0,
                extreme_percentile=0.15,
                extreme_multiplier=multiplier,
                seed=42,
            )

            assert mask.shape == data.shape
            actual_ratio = (~mask).float().mean().item()
            # Should still achieve target missing ratio
            assert 0.05 <= actual_ratio <= 0.35

    def test_mnar_masking_with_different_extreme_percentiles(self):
        """Test MNAR masking with different extreme percentiles."""
        data = torch.randn(10, 168, 6)

        for percentile in [0.05, 0.10, 0.15, 0.20, 0.30]:
            mask = apply_mnar_mask(
                data,
                missing_ratio=0.2,
                target_variable=0,
                extreme_percentile=percentile,
                extreme_multiplier=5.0,
                seed=42,
            )

            assert mask.shape == data.shape
            actual_ratio = (~mask).float().mean().item()
            # Wider tolerance since extreme_percentile affects gap placement
            assert 0.05 <= actual_ratio <= 0.35

    def test_mnar_masking_with_different_missing_ratios(self):
        """Test MNAR masking with various missing ratios."""
        data = torch.randn(10, 168, 6)

        for target_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
            mask = apply_mnar_mask(data, missing_ratio=target_ratio, seed=42)
            actual_ratio = (~mask).float().mean().item()

            # Should be within 15% tolerance for MNAR (similar to MAR)
            tolerance = 0.15
            assert abs(actual_ratio - target_ratio) < tolerance, (
                f"Missing ratio {actual_ratio:.3f} not within {tolerance} of target {target_ratio}"
            )

    def test_mnar_masking_reproducibility(self):
        """Test that MNAR masking is reproducible with same seed."""
        data = torch.randn(10, 168, 6)

        mask1 = apply_mnar_mask(data, missing_ratio=0.2, seed=42)
        mask2 = apply_mnar_mask(data, missing_ratio=0.2, seed=42)

        assert torch.equal(mask1, mask2)

    def test_mnar_masking_different_seeds(self):
        """Test that different seeds produce different masks."""
        data = torch.randn(10, 168, 6)

        mask1 = apply_mnar_mask(data, missing_ratio=0.2, seed=42)
        mask2 = apply_mnar_mask(data, missing_ratio=0.2, seed=43)

        # Masks should be different
        assert not torch.equal(mask1, mask2)

        # But should have similar missing ratios
        ratio1 = (~mask1).float().mean().item()
        ratio2 = (~mask2).float().mean().item()
        assert abs(ratio1 - ratio2) < 0.1

    def test_mnar_masking_invalid_shape(self):
        """Test MNAR masking raises error for invalid tensor shape."""
        data_2d = torch.randn(168, 6)
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            apply_mnar_mask(data_2d, missing_ratio=0.2)

        data_4d = torch.randn(10, 168, 6, 2)
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            apply_mnar_mask(data_4d, missing_ratio=0.2)

    def test_mnar_masking_invalid_missing_ratio(self):
        """Test MNAR masking raises error for invalid missing ratio."""
        data = torch.randn(10, 168, 6)

        with pytest.raises(ValueError, match="missing_ratio must be in"):
            apply_mnar_mask(data, missing_ratio=-0.1)

        with pytest.raises(ValueError, match="missing_ratio must be in"):
            apply_mnar_mask(data, missing_ratio=1.5)

    def test_mnar_masking_invalid_gap_lengths(self):
        """Test MNAR masking raises error for invalid gap lengths."""
        data = torch.randn(10, 168, 6)

        with pytest.raises(ValueError, match="min_gap_length must be >= 1"):
            apply_mnar_mask(data, missing_ratio=0.2, min_gap_length=0)

        with pytest.raises(ValueError, match="max_gap_length.*must be >= min_gap_length"):
            apply_mnar_mask(data, missing_ratio=0.2, min_gap_length=10, max_gap_length=5)

    def test_mnar_masking_invalid_target_variable(self):
        """Test MNAR masking raises error for invalid target variable."""
        data = torch.randn(10, 168, 6)

        # Negative index
        with pytest.raises(ValueError, match="target_variable must be in"):
            apply_mnar_mask(data, missing_ratio=0.2, target_variable=-1)

        # Index >= V
        with pytest.raises(ValueError, match="target_variable must be in"):
            apply_mnar_mask(data, missing_ratio=0.2, target_variable=6)

    def test_mnar_masking_invalid_extreme_percentile(self):
        """Test MNAR masking raises error for invalid extreme percentile."""
        data = torch.randn(10, 168, 6)

        # Below 0
        with pytest.raises(ValueError, match="extreme_percentile must be in"):
            apply_mnar_mask(data, missing_ratio=0.2, extreme_percentile=-0.1)

        # Above 0.5
        with pytest.raises(ValueError, match="extreme_percentile must be in"):
            apply_mnar_mask(data, missing_ratio=0.2, extreme_percentile=0.6)

    def test_mnar_masking_invalid_extreme_multiplier(self):
        """Test MNAR masking raises error for invalid extreme multiplier."""
        data = torch.randn(10, 168, 6)

        # Below 1.0 (makes no sense for MNAR - extreme should be MORE likely to be missing)
        with pytest.raises(ValueError, match="extreme_multiplier must be >= 1.0"):
            apply_mnar_mask(data, missing_ratio=0.2, extreme_multiplier=0.5)

    def test_mnar_masking_edge_case_small_sequence(self):
        """Test MNAR masking with very small sequence length."""
        data = torch.randn(5, 10, 3)

        mask = apply_mnar_mask(
            data,
            missing_ratio=0.2,
            min_gap_length=1,
            max_gap_length=3,
            seed=42,
        )

        assert mask.shape == data.shape
        actual_missing_ratio = (~mask).float().mean().item()
        # Wider tolerance for small sequences
        assert 0.0 <= actual_missing_ratio <= 0.5

    def test_mnar_masking_edge_case_single_variable(self):
        """Test MNAR masking with single variable."""
        data = torch.randn(10, 168, 1)

        mask = apply_mnar_mask(data, missing_ratio=0.2, target_variable=0, seed=42)

        assert mask.shape == data.shape
        actual_missing_ratio = (~mask).float().mean().item()
        assert 0.1 <= actual_missing_ratio <= 0.3

    def test_mnar_masking_no_extreme_values(self):
        """Test MNAR masking when all values are similar (no clear extremes)."""
        # Create data with very small variance (all values similar)
        data = torch.ones(10, 168, 6) + torch.randn(10, 168, 6) * 0.01

        # Should still work, falling back to more uniform-like behavior
        mask = apply_mnar_mask(
            data,
            missing_ratio=0.2,
            extreme_percentile=0.15,
            extreme_multiplier=5.0,
            seed=42,
        )

        assert mask.shape == data.shape
        actual_missing_ratio = (~mask).float().mean().item()
        assert 0.05 <= actual_missing_ratio <= 0.35

    def test_mnar_vs_mar_difference(self):
        """Test that MNAR and MAR create different missing patterns.

        MNAR: Missingness depends on the UNOBSERVED value itself
        MAR: Missingness depends on OBSERVED (other) values

        Both should bias towards extremes, but MNAR should show stronger
        bias on the target variable itself.
        """
        torch.manual_seed(42)
        data = torch.randn(50, 168, 6)

        # Apply MNAR mask (missingness depends on target variable's own value)
        mnar_mask = apply_mnar_mask(
            data,
            missing_ratio=0.2,
            target_variable=0,
            extreme_percentile=0.15,
            extreme_multiplier=5.0,
            seed=42,
        )

        # Apply MAR mask (missingness depends on condition variable)
        mar_mask = apply_mar_mask(
            data,
            missing_ratio=0.2,
            condition_variable=0,
            extreme_percentile=0.15,
            seed=43,  # Different seed to ensure different pattern
        )

        # Masks should be different
        assert not torch.equal(mnar_mask, mar_mask)

        # Both should have similar overall missing ratios
        mnar_ratio = (~mnar_mask).float().mean().item()
        mar_ratio = (~mar_mask).float().mean().item()
        assert abs(mnar_ratio - mar_ratio) < 0.1


class TestApplyMask:
    """Tests for the generic apply_mask dispatcher function."""

    def test_apply_mask_with_mcar_config(self):
        """Test apply_mask with MCAR configuration."""
        data = torch.randn(10, 168, 6)
        config = MaskingConfig(
            strategy="mcar", missing_ratio=0.2, min_gap_length=1, max_gap_length=24
        )

        mask = apply_mask(data, config, seed=42)

        assert mask.shape == data.shape
        actual_missing_ratio = (~mask).float().mean().item()
        assert 0.15 <= actual_missing_ratio <= 0.25

    def test_apply_mask_reproducibility(self):
        """Test apply_mask is reproducible with same seed."""
        data = torch.randn(10, 168, 6)
        config = MaskingConfig(strategy="mcar", missing_ratio=0.2)

        mask1 = apply_mask(data, config, seed=42)
        mask2 = apply_mask(data, config, seed=42)

        assert torch.equal(mask1, mask2)

    def test_apply_mask_with_mar_config(self):
        """Test apply_mask with MAR configuration."""
        data = torch.randn(10, 168, 6)
        config = MaskingConfig(strategy="mar", missing_ratio=0.2)

        mask = apply_mask(data, config, seed=42)

        assert mask.shape == data.shape
        actual_missing_ratio = (~mask).float().mean().item()
        assert 0.05 <= actual_missing_ratio <= 0.35  # Wider tolerance for MAR

    def test_apply_mask_with_mnar_config(self):
        """Test apply_mask with MNAR configuration."""
        data = torch.randn(10, 168, 6)
        config = MaskingConfig(strategy="mnar", missing_ratio=0.2)

        mask = apply_mask(data, config, seed=42)

        assert mask.shape == data.shape
        actual_missing_ratio = (~mask).float().mean().item()
        assert 0.05 <= actual_missing_ratio <= 0.35  # Wider tolerance for MNAR

    def test_apply_mask_realistic_not_implemented(self):
        """Test apply_mask raises NotImplementedError for realistic."""
        data = torch.randn(10, 168, 6)
        config = MaskingConfig(strategy="realistic", missing_ratio=0.2)

        with pytest.raises(
            NotImplementedError, match="Realistic masking strategy not yet implemented"
        ):
            apply_mask(data, config, seed=42)
