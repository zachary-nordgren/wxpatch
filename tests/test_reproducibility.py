"""Tests for reproducibility utilities."""

import random

import numpy as np
import pytest
import torch

from weather_imputation.utils.reproducibility import (
    get_rng_state,
    make_reproducible,
    seed_everything,
    set_rng_state,
)


class TestSeedEverything:
    """Tests for seed_everything function."""

    def test_seed_everything_basic(self):
        """Test that seed_everything sets all RNG seeds."""
        seed_everything(42)

        # Generate some random values
        py_val = random.random()
        np_val = np.random.rand()
        torch_val = torch.rand(1).item()

        # Reset and verify reproducibility
        seed_everything(42)
        assert random.random() == py_val
        assert np.random.rand() == np_val
        assert torch.rand(1).item() == torch_val

    def test_seed_everything_different_seeds(self):
        """Test that different seeds produce different results."""
        seed_everything(42)
        val1 = torch.rand(10)

        seed_everything(123)
        val2 = torch.rand(10)

        # Different seeds should produce different values
        assert not torch.equal(val1, val2)

    def test_seed_everything_deterministic_mode(self):
        """Test deterministic mode configuration."""
        seed_everything(42, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

        seed_everything(42, deterministic=False)
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True

    def test_seed_everything_invalid_seed(self):
        """Test that invalid seeds raise ValueError."""
        with pytest.raises(ValueError, match="Seed must be in range"):
            seed_everything(-1)

        with pytest.raises(ValueError, match="Seed must be in range"):
            seed_everything(2**32)

    def test_seed_everything_boundary_seeds(self):
        """Test boundary seed values."""
        # Minimum valid seed
        seed_everything(0)
        val1 = torch.rand(5)

        # Maximum valid seed
        seed_everything(2**32 - 1)
        val2 = torch.rand(5)

        assert not torch.equal(val1, val2)

    def test_seed_everything_python_random(self):
        """Test Python random module seeding."""
        seed_everything(42)
        py_vals1 = [random.random() for _ in range(10)]

        seed_everything(42)
        py_vals2 = [random.random() for _ in range(10)]

        assert py_vals1 == py_vals2

    def test_seed_everything_numpy_random(self):
        """Test NumPy random seeding."""
        seed_everything(42)
        np_vals1 = np.random.randn(10)

        seed_everything(42)
        np_vals2 = np.random.randn(10)

        np.testing.assert_array_equal(np_vals1, np_vals2)

    def test_seed_everything_torch_random(self):
        """Test PyTorch random seeding."""
        seed_everything(42)
        torch_vals1 = torch.randn(10)

        seed_everything(42)
        torch_vals2 = torch.randn(10)

        assert torch.equal(torch_vals1, torch_vals2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_seed_everything_cuda(self):
        """Test CUDA random seeding."""
        seed_everything(42)
        cuda_vals1 = torch.randn(10, device="cuda")

        seed_everything(42)
        cuda_vals2 = torch.randn(10, device="cuda")

        assert torch.equal(cuda_vals1, cuda_vals2)


class TestRNGStateManagement:
    """Tests for RNG state capture and restoration."""

    def test_get_rng_state_structure(self):
        """Test that get_rng_state returns expected structure."""
        state = get_rng_state()

        assert isinstance(state, dict)
        assert "python" in state
        assert "numpy" in state
        assert "torch" in state
        assert "torch_cuda" in state
        assert "cudnn_deterministic" in state
        assert "cudnn_benchmark" in state

    def test_get_rng_state_cuda_list(self):
        """Test that CUDA states are returned as list."""
        state = get_rng_state()
        assert isinstance(state["torch_cuda"], list)

        if torch.cuda.is_available():
            assert len(state["torch_cuda"]) == torch.cuda.device_count()
        else:
            assert len(state["torch_cuda"]) == 0

    def test_set_rng_state_basic(self):
        """Test basic RNG state restoration."""
        seed_everything(42)
        state = get_rng_state()

        # Advance RNG state
        _ = torch.rand(100)
        _ = np.random.rand(100)
        _ = random.random()

        # Restore state
        set_rng_state(state)

        # Should produce same values as immediately after seed_everything(42)
        seed_everything(42)
        expected_torch = torch.rand(10)
        expected_numpy = np.random.rand(10)
        expected_python = random.random()

        set_rng_state(state)
        assert torch.equal(torch.rand(10), expected_torch)
        np.testing.assert_array_equal(np.random.rand(10), expected_numpy)
        assert random.random() == expected_python

    def test_set_rng_state_missing_keys(self):
        """Test that set_rng_state raises ValueError for incomplete state."""
        incomplete_state = {"python": random.getstate()}

        with pytest.raises(ValueError, match="missing required keys"):
            set_rng_state(incomplete_state)

    def test_rng_state_round_trip(self):
        """Test capturing and restoring state multiple times."""
        seed_everything(42)

        # Capture initial state
        state1 = get_rng_state()
        vals1 = torch.rand(5)

        # Advance state and capture
        state2 = get_rng_state()
        vals2 = torch.rand(5)

        # Restore to state1 and verify
        set_rng_state(state1)
        assert torch.equal(torch.rand(5), vals1)

        # Restore to state2 and verify
        set_rng_state(state2)
        assert torch.equal(torch.rand(5), vals2)

    def test_rng_state_independence(self):
        """Test that different RNG sources are independent."""
        seed_everything(42)
        state = get_rng_state()

        # Advance only PyTorch RNG
        _ = torch.rand(100)

        # Python and NumPy should not be affected
        py_val = random.random()
        np_val = np.random.rand()

        # Restore and check Python and NumPy produce same values
        set_rng_state(state)
        assert random.random() == py_val
        assert np.random.rand() == np_val

    def test_rng_state_cudnn_settings(self):
        """Test that CuDNN settings are preserved."""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        state = get_rng_state()

        # Change settings
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        # Restore
        set_rng_state(state)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


class TestMakeReproducibleDecorator:
    """Tests for make_reproducible decorator."""

    def test_make_reproducible_basic(self):
        """Test basic decorator functionality."""

        @make_reproducible
        def generate_data(n: int, seed: int = 42):
            return torch.randn(n)

        data1 = generate_data(10, seed=42)
        data2 = generate_data(10, seed=42)

        assert torch.equal(data1, data2)

    def test_make_reproducible_default_seed(self):
        """Test decorator with default seed."""

        @make_reproducible
        def generate_data(n: int):
            return torch.randn(n)

        data1 = generate_data(10)
        data2 = generate_data(10)

        assert torch.equal(data1, data2)

    def test_make_reproducible_different_seeds(self):
        """Test decorator produces different results with different seeds."""

        @make_reproducible
        def generate_data(n: int, seed: int = 42):
            return torch.randn(n)

        data1 = generate_data(10, seed=42)
        data2 = generate_data(10, seed=123)

        assert not torch.equal(data1, data2)

    def test_make_reproducible_with_args(self):
        """Test decorator with multiple arguments."""

        @make_reproducible
        def generate_matrix(rows: int, cols: int, seed: int = 42):
            return torch.randn(rows, cols)

        mat1 = generate_matrix(5, 3, seed=42)
        mat2 = generate_matrix(5, 3, seed=42)

        assert torch.equal(mat1, mat2)
        assert mat1.shape == (5, 3)


class TestReproducibilityIntegration:
    """Integration tests for reproducibility across components."""

    def test_reproducible_data_loading_simulation(self):
        """Test reproducible random masking (simulates data loading)."""
        seed_everything(42)

        # Simulate random masking
        data = torch.randn(100, 10)
        mask1 = torch.rand_like(data) > 0.3

        seed_everything(42)
        data = torch.randn(100, 10)
        mask2 = torch.rand_like(data) > 0.3

        assert torch.equal(mask1, mask2)

    def test_reproducible_model_initialization(self):
        """Test reproducible model initialization."""
        seed_everything(42)
        layer1 = torch.nn.Linear(10, 5)
        weights1 = layer1.weight.clone()

        seed_everything(42)
        layer2 = torch.nn.Linear(10, 5)
        weights2 = layer2.weight.clone()

        assert torch.equal(weights1, weights2)

    def test_reproducible_training_step(self):
        """Test reproducible forward pass and loss computation."""
        seed_everything(42)

        # Simple model and data
        model = torch.nn.Linear(10, 1)
        data = torch.randn(32, 10)
        target = torch.randn(32, 1)

        # Forward pass
        output1 = model(data)
        loss1 = torch.nn.functional.mse_loss(output1, target)

        # Reset and repeat
        seed_everything(42)
        model = torch.nn.Linear(10, 1)
        data = torch.randn(32, 10)
        target = torch.randn(32, 1)

        output2 = model(data)
        loss2 = torch.nn.functional.mse_loss(output2, target)

        assert torch.equal(output1, output2)
        assert loss1.item() == loss2.item()

    def test_reproducible_data_augmentation(self):
        """Test reproducible random data augmentation."""
        seed_everything(42)

        # Simulate augmentation with random noise
        data = torch.ones(10, 5)
        noise1 = torch.randn_like(data) * 0.1
        augmented1 = data + noise1

        seed_everything(42)
        data = torch.ones(10, 5)
        noise2 = torch.randn_like(data) * 0.1
        augmented2 = data + noise2

        assert torch.equal(augmented1, augmented2)

    def test_state_preservation_across_epochs(self):
        """Test that RNG state can be saved and restored across epochs."""
        seed_everything(42)

        # Simulate epoch 1
        _ = torch.randn(100)  # epoch 1 data
        epoch1_state = get_rng_state()

        # Simulate epoch 2
        epoch2_data = torch.randn(100)
        epoch2_state = get_rng_state()

        # Restore to epoch 1 and verify
        set_rng_state(epoch1_state)
        assert torch.equal(torch.randn(100), epoch2_data)

        # Restore to epoch 2 and verify next value
        set_rng_state(epoch2_state)
        epoch3_data = torch.randn(100)

        # Reset and simulate fresh run
        seed_everything(42)
        _ = torch.randn(100)  # epoch 1
        _ = torch.randn(100)  # epoch 2
        expected_epoch3 = torch.randn(100)  # epoch 3

        assert torch.equal(epoch3_data, expected_epoch3)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_multiple_seed_everything_calls(self):
        """Test multiple consecutive seed_everything calls."""
        seed_everything(42)
        val1 = torch.rand(5)

        seed_everything(42)
        val2 = torch.rand(5)

        seed_everything(42)
        val3 = torch.rand(5)

        assert torch.equal(val1, val2)
        assert torch.equal(val2, val3)

    def test_seed_everything_with_operations(self):
        """Test seed_everything after various operations."""
        # Create some tensors and do operations
        _ = torch.randn(1000, 1000)
        _ = torch.nn.functional.conv2d(
            torch.randn(1, 1, 28, 28), torch.randn(1, 1, 3, 3)
        )

        # Seed should still work
        seed_everything(42)
        val1 = torch.rand(10)

        seed_everything(42)
        val2 = torch.rand(10)

        assert torch.equal(val1, val2)

    def test_rng_state_with_empty_operations(self):
        """Test state capture without any RNG operations."""
        seed_everything(42)
        state1 = get_rng_state()

        # Capture again immediately
        state2 = get_rng_state()

        # States should be identical (same RNG position)
        # Note: We can't directly compare state dicts due to different types
        # So we verify they produce same results
        set_rng_state(state1)
        val1 = torch.rand(5)

        set_rng_state(state2)
        val2 = torch.rand(5)

        assert torch.equal(val1, val2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_state_with_multiple_devices(self):
        """Test CUDA state capture with multiple devices."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multiple CUDA devices not available")

        seed_everything(42)

        # Generate data on different devices
        vals_device0 = torch.rand(10, device="cuda:0")
        vals_device1 = torch.rand(10, device="cuda:1")

        # Capture state
        state = get_rng_state()

        # Advance RNG
        _ = torch.rand(100, device="cuda:0")
        _ = torch.rand(100, device="cuda:1")

        # Restore and verify
        set_rng_state(state)
        assert torch.equal(torch.rand(10, device="cuda:0"), vals_device0)
        assert torch.equal(torch.rand(10, device="cuda:1"), vals_device1)
