"""Tests for circular statistics utilities."""

import math

import pytest
import torch

from weather_imputation.utils.circular import (
    angular_difference,
    circular_mean,
    circular_std,
    decode_wind_direction,
    degrees_to_radians,
    encode_wind_direction,
    radians_to_degrees,
)


class TestDegreesRadiansConversion:
    """Test degree/radian conversion functions."""

    def test_degrees_to_radians_cardinal_directions(self):
        """Test conversion for cardinal directions."""
        degrees = torch.tensor([0.0, 90.0, 180.0, 270.0])
        radians = degrees_to_radians(degrees)
        expected = torch.tensor([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
        torch.testing.assert_close(radians, expected, rtol=1e-5, atol=1e-7)

    def test_radians_to_degrees_cardinal_directions(self):
        """Test conversion for cardinal directions."""
        radians = torch.tensor([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
        degrees = radians_to_degrees(radians)
        expected = torch.tensor([0.0, 90.0, 180.0, 270.0])
        torch.testing.assert_close(degrees, expected, rtol=1e-5, atol=1e-7)

    def test_round_trip_conversion(self):
        """Test that degrees -> radians -> degrees is identity."""
        original = torch.tensor([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
        radians = degrees_to_radians(original)
        recovered = radians_to_degrees(radians)
        torch.testing.assert_close(recovered, original, rtol=1e-5, atol=1e-7)

    def test_normalization_to_360(self):
        """Test that radians_to_degrees normalizes to [0, 360)."""
        # Angles beyond [0, 2π] should be normalized
        radians = torch.tensor([0.0, 2 * math.pi, 3 * math.pi, -math.pi])
        degrees = radians_to_degrees(radians)
        expected = torch.tensor([0.0, 0.0, 180.0, 180.0])
        torch.testing.assert_close(degrees, expected, rtol=1e-5, atol=1e-7)


class TestWindDirectionEncoding:
    """Test wind direction encoding/decoding."""

    def test_encode_cardinal_directions(self):
        """Test encoding of cardinal directions."""
        degrees = torch.tensor([0.0, 90.0, 180.0, 270.0])
        encoded = encode_wind_direction(degrees)

        assert encoded.shape == (4, 2)

        # North (0°): (sin=0, cos=1)
        torch.testing.assert_close(encoded[0], torch.tensor([0.0, 1.0]), rtol=1e-5, atol=1e-7)
        # East (90°): (sin=1, cos=0)
        torch.testing.assert_close(encoded[1], torch.tensor([1.0, 0.0]), rtol=1e-5, atol=1e-7)
        # South (180°): (sin=0, cos=-1)
        torch.testing.assert_close(encoded[2], torch.tensor([0.0, -1.0]), rtol=1e-5, atol=1e-7)
        # West (270°): (sin=-1, cos=0)
        torch.testing.assert_close(
            encoded[3], torch.tensor([-1.0, 0.0]), rtol=1e-5, atol=1e-7
        )

    def test_encode_multidimensional(self):
        """Test encoding preserves shape except last dim becomes 2."""
        degrees = torch.randn(3, 4, 5) * 360.0  # Shape: (3, 4, 5)
        encoded = encode_wind_direction(degrees)
        assert encoded.shape == (3, 4, 5, 2)

    def test_encode_boundary_continuity(self):
        """Test encoding is continuous at 0°/360° boundary."""
        degrees = torch.tensor([0.0, 360.0, -0.001, 360.001])
        encoded = encode_wind_direction(degrees)

        # All should encode to approximately (sin=0, cos=1)
        for i in range(4):
            torch.testing.assert_close(
                encoded[i], torch.tensor([0.0, 1.0]), rtol=1e-2, atol=1e-2
            )

    def test_encode_unknown_method_raises_error(self):
        """Test that unknown encoding method raises ValueError."""
        degrees = torch.tensor([0.0, 90.0])
        with pytest.raises(ValueError, match="Unknown encoding method"):
            encode_wind_direction(degrees, method="unknown")  # type: ignore

    def test_decode_cardinal_directions(self):
        """Test decoding of cardinal directions."""
        # (sin, cos) pairs for cardinal directions
        encoded = torch.tensor(
            [
                [0.0, 1.0],  # North (0°)
                [1.0, 0.0],  # East (90°)
                [0.0, -1.0],  # South (180°)
                [-1.0, 0.0],  # West (270°)
            ]
        )
        degrees = decode_wind_direction(encoded)
        expected = torch.tensor([0.0, 90.0, 180.0, 270.0])
        torch.testing.assert_close(degrees, expected, rtol=1e-5, atol=1e-7)

    def test_decode_multidimensional(self):
        """Test decoding preserves shape except last dim is removed."""
        encoded = torch.randn(3, 4, 5, 2)
        # Normalize to unit vectors
        encoded = encoded / torch.norm(encoded, dim=-1, keepdim=True)
        degrees = decode_wind_direction(encoded)
        assert degrees.shape == (3, 4, 5)

    def test_decode_invalid_shape_raises_error(self):
        """Test that invalid encoded shape raises ValueError."""
        encoded = torch.randn(4, 3)  # Last dim should be 2
        with pytest.raises(ValueError, match="Expected last dimension to be 2"):
            decode_wind_direction(encoded)

    def test_encode_decode_round_trip(self):
        """Test that encode -> decode is identity."""
        original = torch.tensor([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
        encoded = encode_wind_direction(original)
        decoded = decode_wind_direction(encoded)
        torch.testing.assert_close(decoded, original, rtol=1e-5, atol=1e-7)

    def test_encode_decode_with_noise(self):
        """Test encoding/decoding with random angles."""
        original = torch.rand(100) * 360.0
        encoded = encode_wind_direction(original)
        decoded = decode_wind_direction(encoded)
        torch.testing.assert_close(decoded, original, rtol=1e-4, atol=1e-5)


class TestCircularMean:
    """Test circular mean computation."""

    def test_circular_mean_same_direction(self):
        """Test mean of identical directions."""
        degrees = torch.tensor([45.0, 45.0, 45.0])
        mean = circular_mean(degrees)
        torch.testing.assert_close(mean, torch.tensor(45.0), rtol=1e-5, atol=1e-7)

    def test_circular_mean_cardinal_directions(self):
        """Test mean of cardinal directions."""
        # Mean of North (0°) and East (90°) is Northeast (45°)
        degrees = torch.tensor([0.0, 90.0])
        mean = circular_mean(degrees)
        torch.testing.assert_close(mean, torch.tensor(45.0), rtol=1e-5, atol=1e-7)

    def test_circular_mean_across_zero(self):
        """Test mean of directions across 0°/360° boundary."""
        # Mean of North (0°) and slightly west (350°) is 355°
        degrees = torch.tensor([0.0, 350.0])
        mean = circular_mean(degrees)
        torch.testing.assert_close(mean, torch.tensor(355.0), rtol=1e-5, atol=1e-7)

    def test_circular_mean_opposite_directions(self):
        """Test mean of opposite directions (should be perpendicular)."""
        # Mean of North (0°) and South (180°) is undefined mathematically,
        # but atan2 will return a perpendicular angle (90° or 270°)
        degrees = torch.tensor([0.0, 180.0])
        mean = circular_mean(degrees)
        # Either 90° or 270° is acceptable (perpendicular to both)
        assert torch.isclose(mean, torch.tensor(90.0), atol=1.0) or torch.isclose(
            mean, torch.tensor(270.0), atol=1.0
        )

    def test_circular_mean_four_cardinal_directions(self):
        """Test mean of all four cardinal directions."""
        # Mean of N, E, S, W is undefined (all cancel out)
        degrees = torch.tensor([0.0, 90.0, 180.0, 270.0])
        mean = circular_mean(degrees)
        # Result is unstable, but should be finite
        assert torch.isfinite(mean)

    def test_circular_mean_with_mask(self):
        """Test circular mean with boolean mask."""
        degrees = torch.tensor([0.0, 90.0, 180.0, 270.0])
        mask = torch.tensor([True, True, False, False])  # Only use N and E
        mean = circular_mean(degrees, mask=mask)
        # Mean of North (0°) and East (90°) is Northeast (45°)
        torch.testing.assert_close(mean, torch.tensor(45.0), rtol=1e-5, atol=1e-7)

    def test_circular_mean_multidimensional(self):
        """Test circular mean along last dimension."""
        # Shape: (2, 3) - compute mean over last dim
        degrees = torch.tensor([[0.0, 90.0, 180.0], [45.0, 45.0, 45.0]])
        mean = circular_mean(degrees)
        assert mean.shape == (2,)
        # First row: mean of N, E, S (should be ~90° or 270°)
        # Second row: mean of NE, NE, NE = 45°
        torch.testing.assert_close(mean[1], torch.tensor(45.0), rtol=1e-5, atol=1e-7)

    def test_circular_mean_preserves_dtype(self):
        """Test that circular mean preserves tensor dtype."""
        degrees = torch.tensor([0.0, 90.0], dtype=torch.float32)
        mean = circular_mean(degrees)
        assert mean.dtype == torch.float32


class TestCircularStd:
    """Test circular standard deviation computation."""

    def test_circular_std_same_direction(self):
        """Test std of identical directions is near zero."""
        degrees = torch.tensor([45.0, 45.0, 45.0])
        std = circular_std(degrees)
        # Should be very close to zero
        torch.testing.assert_close(std, torch.tensor(0.0), rtol=1e-3, atol=1e-6)

    def test_circular_std_concentrated_directions(self):
        """Test std of concentrated directions."""
        degrees = torch.tensor([0.0, 1.0, 359.0])
        std = circular_std(degrees)
        # Should be small (concentrated around 0°)
        assert std < 0.1

    def test_circular_std_dispersed_directions(self):
        """Test std of dispersed directions."""
        degrees = torch.tensor([0.0, 90.0, 180.0, 270.0])
        std = circular_std(degrees)
        # Should be large (uniformly distributed)
        assert std > 1.0

    def test_circular_std_opposite_directions(self):
        """Test std of opposite directions."""
        degrees = torch.tensor([0.0, 180.0])
        std = circular_std(degrees)
        # Should be high (maximum dispersion for 2 points)
        assert std > 1.0

    def test_circular_std_with_mask(self):
        """Test circular std with boolean mask."""
        degrees = torch.tensor([0.0, 90.0, 180.0, 270.0])
        mask = torch.tensor([True, True, True, False])  # Exclude West (270°)
        std = circular_std(degrees, mask=mask)
        # Std of N, E, S
        assert std > 0.5  # Should be moderately dispersed

    def test_circular_std_multidimensional(self):
        """Test circular std along last dimension."""
        degrees = torch.tensor([[0.0, 1.0, 359.0], [0.0, 90.0, 180.0]])
        std = circular_std(degrees)
        assert std.shape == (2,)
        # First row: concentrated (low std)
        assert std[0] < 0.1
        # Second row: dispersed (high std)
        assert std[1] > 0.5

    def test_circular_std_preserves_dtype(self):
        """Test that circular std preserves tensor dtype."""
        degrees = torch.tensor([0.0, 90.0], dtype=torch.float32)
        std = circular_std(degrees)
        assert std.dtype == torch.float32


class TestAngularDifference:
    """Test angular difference computation."""

    def test_angular_difference_same_direction(self):
        """Test difference of identical directions is zero."""
        diff = angular_difference(torch.tensor(45.0), torch.tensor(45.0))
        torch.testing.assert_close(diff, torch.tensor(0.0), rtol=1e-5, atol=1e-7)

    def test_angular_difference_clockwise(self):
        """Test clockwise difference is positive."""
        diff = angular_difference(torch.tensor(10.0), torch.tensor(20.0))
        torch.testing.assert_close(diff, torch.tensor(10.0), rtol=1e-5, atol=1e-7)

    def test_angular_difference_counterclockwise(self):
        """Test counterclockwise difference is negative."""
        diff = angular_difference(torch.tensor(20.0), torch.tensor(10.0))
        torch.testing.assert_close(diff, torch.tensor(-10.0), rtol=1e-5, atol=1e-7)

    def test_angular_difference_across_zero_clockwise(self):
        """Test clockwise difference across 0°/360° boundary."""
        diff = angular_difference(torch.tensor(350.0), torch.tensor(10.0))
        torch.testing.assert_close(diff, torch.tensor(20.0), rtol=1e-5, atol=1e-7)

    def test_angular_difference_across_zero_counterclockwise(self):
        """Test counterclockwise difference across 0°/360° boundary."""
        diff = angular_difference(torch.tensor(10.0), torch.tensor(350.0))
        torch.testing.assert_close(diff, torch.tensor(-20.0), rtol=1e-5, atol=1e-7)

    def test_angular_difference_opposite_directions(self):
        """Test difference of opposite directions is ±180°."""
        diff1 = angular_difference(torch.tensor(0.0), torch.tensor(180.0))
        diff2 = angular_difference(torch.tensor(180.0), torch.tensor(0.0))
        # Should be ±180° (both normalize to -180° due to modulo arithmetic)
        torch.testing.assert_close(diff1, torch.tensor(-180.0), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(diff2, torch.tensor(-180.0), rtol=1e-5, atol=1e-7)

    def test_angular_difference_multidimensional(self):
        """Test angular difference with multidimensional tensors."""
        degrees1 = torch.tensor([0.0, 90.0, 180.0])
        degrees2 = torch.tensor([90.0, 180.0, 270.0])
        diff = angular_difference(degrees1, degrees2)
        expected = torch.tensor([90.0, 90.0, 90.0])
        torch.testing.assert_close(diff, expected, rtol=1e-5, atol=1e-7)

    def test_angular_difference_broadcast(self):
        """Test angular difference with broadcasting."""
        degrees1 = torch.tensor([0.0])  # Shape: (1,)
        degrees2 = torch.tensor([90.0, 180.0, 270.0])  # Shape: (3,)
        diff = angular_difference(degrees1, degrees2)
        # 0->90: +90, 0->180: -180 (shorter path), 0->270: -90 (shorter path)
        expected = torch.tensor([90.0, -180.0, -90.0])
        torch.testing.assert_close(diff, expected, rtol=1e-5, atol=1e-7)

    def test_angular_difference_range(self):
        """Test that angular difference is always in [-180, 180]."""
        degrees1 = torch.rand(100) * 360.0
        degrees2 = torch.rand(100) * 360.0
        diff = angular_difference(degrees1, degrees2)
        assert torch.all(diff >= -180.0)
        assert torch.all(diff <= 180.0)

    def test_angular_difference_preserves_dtype(self):
        """Test that angular difference preserves tensor dtype."""
        degrees1 = torch.tensor(0.0, dtype=torch.float32)
        degrees2 = torch.tensor(90.0, dtype=torch.float32)
        diff = angular_difference(degrees1, degrees2)
        assert diff.dtype == torch.float32
