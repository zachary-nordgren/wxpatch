"""Utilities for ensuring reproducible experiments.

This module provides functions for managing random number generator (RNG) state
across Python's random module, NumPy, and PyTorch to ensure bit-exact reproducibility.

References:
    - PyTorch Reproducibility Guide: https://pytorch.org/docs/stable/notes/randomness.html
    - NFR-006 from SPEC.md: "Bit-exact results with same seed"
"""

import logging
import random
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for Python's random module, NumPy, and PyTorch (both CPU and CUDA).
    Optionally enables PyTorch's deterministic mode for bit-exact reproducibility.

    Args:
        seed: Random seed value (should be in range [0, 2^32 - 1])
        deterministic: If True, enable PyTorch deterministic operations.
            This may reduce performance but ensures reproducibility.
            When False, some operations may use non-deterministic algorithms
            for better performance.

    Example:
        >>> from weather_imputation.utils.reproducibility import seed_everything
        >>> seed_everything(42)  # Set all seeds to 42
        >>> # Now all random operations will be reproducible

    Note:
        Even with all seeds set and deterministic=True, some PyTorch operations
        on CUDA may still be non-deterministic due to hardware limitations.
        For truly deterministic behavior, consider using CPU-only or specific
        CUDA configurations.

    References:
        - PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    """
    # Validate seed range
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError(f"Seed must be in range [0, 2^32 - 1], got {seed}")

    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed (CPU and CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Configure deterministic behavior
    if deterministic:
        # Enable deterministic mode (may reduce performance)
        torch.use_deterministic_algorithms(True)
        # Set CuDNN to deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug(
            f"Seed set to {seed} with deterministic mode enabled "
            "(may impact performance)"
        )
    else:
        # Allow non-deterministic but faster operations
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # Auto-tune for performance
        logger.debug(
            f"Seed set to {seed} with non-deterministic mode "
            "(better performance, less reproducible)"
        )


def get_rng_state() -> dict[str, Any]:
    """Capture current RNG state from all sources.

    Returns a dictionary containing the current state of all random number
    generators (Python random, NumPy, PyTorch CPU, PyTorch CUDA).

    Returns:
        Dictionary with keys:
            - 'python': Python random module state
            - 'numpy': NumPy random state
            - 'torch': PyTorch CPU RNG state
            - 'torch_cuda': List of PyTorch CUDA RNG states (one per device)
            - 'cudnn_deterministic': CuDNN deterministic flag
            - 'cudnn_benchmark': CuDNN benchmark flag

    Example:
        >>> state = get_rng_state()
        >>> # ... perform some random operations ...
        >>> set_rng_state(state)  # Restore to previous state

    Note:
        The returned state can be large (especially CUDA states) and should
        be stored carefully in checkpoints.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }

    # Capture CUDA RNG states if available
    if torch.cuda.is_available():
        # Get state for all CUDA devices
        cuda_states = []
        for device_idx in range(torch.cuda.device_count()):
            with torch.cuda.device(device_idx):
                cuda_states.append(torch.cuda.get_rng_state())
        state["torch_cuda"] = cuda_states
    else:
        state["torch_cuda"] = []

    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state from a previously captured state.

    Restores the state of all random number generators to a previously
    captured state (from get_rng_state()).

    Args:
        state: Dictionary returned by get_rng_state()

    Raises:
        ValueError: If state dictionary is missing required keys
        RuntimeError: If CUDA state restoration fails

    Example:
        >>> state = get_rng_state()
        >>> # ... perform some random operations ...
        >>> set_rng_state(state)  # Restore to previous state
        >>> # Random operations will now continue from the saved state

    Note:
        If the state was captured on a different number of CUDA devices,
        this function will attempt to restore as many states as possible
        but will warn about mismatches.
    """
    # Validate state dictionary
    required_keys = {
        "python",
        "numpy",
        "torch",
        "torch_cuda",
        "cudnn_deterministic",
        "cudnn_benchmark",
    }
    missing_keys = required_keys - set(state.keys())
    if missing_keys:
        raise ValueError(f"State dictionary missing required keys: {missing_keys}")

    # Restore Python random state
    random.setstate(state["python"])

    # Restore NumPy random state
    np.random.set_state(state["numpy"])

    # Restore PyTorch CPU RNG state
    torch.set_rng_state(state["torch"])

    # Restore PyTorch CUDA RNG states
    cuda_states = state["torch_cuda"]
    if torch.cuda.is_available() and cuda_states:
        current_device_count = torch.cuda.device_count()
        saved_device_count = len(cuda_states)

        if current_device_count != saved_device_count:
            logger.warning(
                f"Device count mismatch: current={current_device_count}, "
                f"saved={saved_device_count}. Restoring available states."
            )

        # Restore states for available devices
        for device_idx in range(min(current_device_count, saved_device_count)):
            with torch.cuda.device(device_idx):
                torch.cuda.set_rng_state(cuda_states[device_idx])

    # Restore CuDNN settings
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]

    logger.debug("RNG state restored successfully")


def make_reproducible(func):
    """Decorator to ensure a function runs with a specific seed.

    This decorator can be used to wrap functions that should produce
    deterministic results. The seed is passed as a keyword argument.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that sets RNG state before execution

    Example:
        >>> @make_reproducible
        ... def generate_random_data(n: int, seed: int = 42):
        ...     return torch.randn(n)
        >>>
        >>> data1 = generate_random_data(10, seed=42)
        >>> data2 = generate_random_data(10, seed=42)
        >>> assert torch.equal(data1, data2)  # Bit-exact match

    Note:
        The function must accept a 'seed' keyword argument.
        If not provided, defaults to 42.
    """

    def wrapper(*args, **kwargs):
        seed = kwargs.pop("seed", 42)
        seed_everything(seed, deterministic=True)
        return func(*args, **kwargs)

    return wrapper
