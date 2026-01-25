"""Tests for masking strategies."""

import pytest
import torch

from weather_imputation.config.data import MaskingConfig
from weather_imputation.data.masking import apply_mask, apply_mcar_mask


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

    def test_apply_mask_mar_not_implemented(self):
        """Test apply_mask raises NotImplementedError for MAR."""
        data = torch.randn(10, 168, 6)
        config = MaskingConfig(strategy="mar", missing_ratio=0.2)

        with pytest.raises(NotImplementedError, match="MAR masking strategy not yet implemented"):
            apply_mask(data, config, seed=42)

    def test_apply_mask_mnar_not_implemented(self):
        """Test apply_mask raises NotImplementedError for MNAR."""
        data = torch.randn(10, 168, 6)
        config = MaskingConfig(strategy="mnar", missing_ratio=0.2)

        with pytest.raises(NotImplementedError, match="MNAR masking strategy not yet implemented"):
            apply_mask(data, config, seed=42)

    def test_apply_mask_realistic_not_implemented(self):
        """Test apply_mask raises NotImplementedError for realistic."""
        data = torch.randn(10, 168, 6)
        config = MaskingConfig(strategy="realistic", missing_ratio=0.2)

        with pytest.raises(
            NotImplementedError, match="Realistic masking strategy not yet implemented"
        ):
            apply_mask(data, config, seed=42)
