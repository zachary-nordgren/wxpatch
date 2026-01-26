"""Tests for base imputation protocol and base class."""

from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from weather_imputation.models.base import BaseImputer, Imputer

# =============================================================================
# Mock Implementations for Testing
# =============================================================================


class MockImputer(BaseImputer):
    """Mock imputer implementing the Imputer protocol for testing."""

    def __init__(self, name: str = "MockImputer", param1: float = 1.0):
        super().__init__(name)
        self._hyperparameters = {"param1": param1}
        self.fit_called = False
        self.impute_called = False
        self.save_called = False
        self.load_called = False

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        """Mock fit method."""
        self.fit_called = True
        self._is_fitted = True

    def impute(self, observed: Tensor, mask: Tensor) -> Tensor:
        """Mock impute method - returns observed values unchanged."""
        self._check_fitted()
        self._validate_inputs(observed, mask)
        self.impute_called = True
        return observed.clone()

    def save(self, path: Path) -> None:
        """Mock save method."""
        self._save_metadata(path)
        self.save_called = True

    def load(self, path: Path) -> None:
        """Mock load method."""
        metadata = self._load_metadata(path)
        self.name = metadata["name"]
        self._is_fitted = metadata["is_fitted"]
        self._hyperparameters = metadata["hyperparameters"]
        self.load_called = True


class IncompatibleClass:
    """Class that doesn't implement the Imputer protocol."""

    def some_method(self) -> None:
        pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_imputer() -> MockImputer:
    """Create a mock imputer instance."""
    return MockImputer(name="TestImputer", param1=2.5)


@pytest.fixture
def sample_data() -> tuple[Tensor, Tensor]:
    """Create sample observed data and mask.

    Returns:
        Tuple of (observed, mask) tensors.
        - observed: (2, 10, 3) tensor with some values
        - mask: (2, 10, 3) boolean tensor
    """
    torch.manual_seed(42)
    observed = torch.randn(2, 10, 3)
    mask = torch.rand(2, 10, 3) > 0.3  # ~70% observed
    # Set masked positions to 0 in observed
    observed = observed * mask.float()
    return observed, mask


@pytest.fixture
def sample_dataloader(sample_data: tuple[Tensor, Tensor]) -> DataLoader:
    """Create a sample DataLoader for testing fit()."""
    observed, mask = sample_data
    target = observed.clone()
    dataset = TensorDataset(observed, mask, target)
    return DataLoader(dataset, batch_size=2)


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


def test_protocol_compliance() -> None:
    """Test that MockImputer correctly implements the Imputer protocol."""
    imputer = MockImputer()
    assert isinstance(imputer, Imputer)


def test_protocol_non_compliance() -> None:
    """Test that incompatible classes don't satisfy the protocol."""
    obj = IncompatibleClass()
    assert not isinstance(obj, Imputer)


def test_protocol_has_required_methods() -> None:
    """Test that Imputer protocol defines all required methods."""
    required_methods = {"fit", "impute", "save", "load"}
    protocol_methods = {
        name for name in dir(Imputer) if not name.startswith("_")
    }
    assert required_methods.issubset(protocol_methods)


# =============================================================================
# BaseImputer Initialization Tests
# =============================================================================


def test_base_imputer_initialization() -> None:
    """Test BaseImputer initialization."""
    imputer = MockImputer(name="TestModel", param1=1.5)
    assert imputer.name == "TestModel"
    assert not imputer.is_fitted
    assert imputer.get_hyperparameters() == {"param1": 1.5}


def test_base_imputer_default_name() -> None:
    """Test BaseImputer with default name."""
    imputer = MockImputer()
    assert imputer.name == "MockImputer"


# =============================================================================
# Fitted State Tests
# =============================================================================


def test_is_fitted_initially_false() -> None:
    """Test that is_fitted is False before fit()."""
    imputer = MockImputer()
    assert not imputer.is_fitted


def test_is_fitted_after_fit(
    mock_imputer: MockImputer, sample_dataloader: DataLoader
) -> None:
    """Test that is_fitted becomes True after fit()."""
    mock_imputer.fit(sample_dataloader)
    assert mock_imputer.is_fitted


def test_check_fitted_raises_when_not_fitted(
    mock_imputer: MockImputer, sample_data: tuple[Tensor, Tensor]
) -> None:
    """Test that impute() raises error when model not fitted."""
    observed, mask = sample_data
    with pytest.raises(RuntimeError, match="has not been fitted yet"):
        mock_imputer.impute(observed, mask)


def test_check_fitted_passes_when_fitted(
    mock_imputer: MockImputer,
    sample_dataloader: DataLoader,
    sample_data: tuple[Tensor, Tensor],
) -> None:
    """Test that impute() works after fit()."""
    observed, mask = sample_data
    mock_imputer.fit(sample_dataloader)
    result = mock_imputer.impute(observed, mask)
    assert result.shape == observed.shape


# =============================================================================
# Input Validation Tests
# =============================================================================


def test_validate_inputs_valid_tensors(
    mock_imputer: MockImputer, sample_data: tuple[Tensor, Tensor]
) -> None:
    """Test that valid inputs pass validation."""
    observed, mask = sample_data
    # Should not raise
    mock_imputer._validate_inputs(observed, mask)


def test_validate_inputs_wrong_observed_type(mock_imputer: MockImputer) -> None:
    """Test that non-tensor observed raises ValueError."""
    mask = torch.rand(2, 10, 3) > 0.5
    with pytest.raises(ValueError, match="observed must be a torch.Tensor"):
        mock_imputer._validate_inputs([1, 2, 3], mask)  # type: ignore


def test_validate_inputs_wrong_mask_type(mock_imputer: MockImputer) -> None:
    """Test that non-tensor mask raises ValueError."""
    observed = torch.randn(2, 10, 3)
    with pytest.raises(ValueError, match="mask must be a torch.Tensor"):
        mock_imputer._validate_inputs(observed, [True, False])  # type: ignore


def test_validate_inputs_wrong_observed_dimensions(
    mock_imputer: MockImputer,
) -> None:
    """Test that 2D observed raises ValueError."""
    observed = torch.randn(10, 3)  # 2D instead of 3D
    mask = torch.rand(2, 10, 3) > 0.5
    with pytest.raises(ValueError, match="observed must be 3D"):
        mock_imputer._validate_inputs(observed, mask)


def test_validate_inputs_wrong_mask_dimensions(mock_imputer: MockImputer) -> None:
    """Test that 2D mask raises ValueError."""
    observed = torch.randn(2, 10, 3)
    mask = torch.rand(10, 3) > 0.5  # 2D instead of 3D
    with pytest.raises(ValueError, match="mask must be 3D"):
        mock_imputer._validate_inputs(observed, mask)


def test_validate_inputs_shape_mismatch(mock_imputer: MockImputer) -> None:
    """Test that shape mismatch raises ValueError."""
    observed = torch.randn(2, 10, 3)
    mask = torch.rand(2, 8, 3) > 0.5  # Different time dimension
    with pytest.raises(ValueError, match="must have same shape"):
        mock_imputer._validate_inputs(observed, mask)


def test_validate_inputs_wrong_mask_dtype(mock_imputer: MockImputer) -> None:
    """Test that non-boolean mask raises ValueError."""
    observed = torch.randn(2, 10, 3)
    mask = torch.rand(2, 10, 3)  # float instead of bool
    with pytest.raises(ValueError, match="mask must be boolean tensor"):
        mock_imputer._validate_inputs(observed, mask)


# =============================================================================
# Hyperparameter Tests
# =============================================================================


def test_get_hyperparameters(mock_imputer: MockImputer) -> None:
    """Test getting hyperparameters."""
    params = mock_imputer.get_hyperparameters()
    assert params == {"param1": 2.5}


def test_get_hyperparameters_returns_copy(mock_imputer: MockImputer) -> None:
    """Test that get_hyperparameters returns a copy."""
    params1 = mock_imputer.get_hyperparameters()
    params1["new_param"] = 99
    params2 = mock_imputer.get_hyperparameters()
    assert "new_param" not in params2
    assert params2 == {"param1": 2.5}


# =============================================================================
# Save/Load Tests
# =============================================================================


def test_save_metadata(mock_imputer: MockImputer, tmp_path: Path) -> None:
    """Test saving model metadata."""
    save_path = tmp_path / "model"
    mock_imputer._is_fitted = True
    mock_imputer._save_metadata(save_path)

    assert save_path.exists()
    assert (save_path / "metadata.pt").exists()


def test_save_creates_directory(mock_imputer: MockImputer, tmp_path: Path) -> None:
    """Test that save creates directory if it doesn't exist."""
    save_path = tmp_path / "nested" / "path" / "model"
    assert not save_path.exists()

    mock_imputer._save_metadata(save_path)
    assert save_path.exists()
    assert (save_path / "metadata.pt").exists()


def test_load_metadata(mock_imputer: MockImputer, tmp_path: Path) -> None:
    """Test loading model metadata."""
    save_path = tmp_path / "model"
    mock_imputer._is_fitted = True
    mock_imputer._save_metadata(save_path)

    metadata = mock_imputer._load_metadata(save_path)
    assert metadata["name"] == "TestImputer"
    assert metadata["is_fitted"] is True
    assert metadata["hyperparameters"] == {"param1": 2.5}


def test_load_metadata_not_found(mock_imputer: MockImputer, tmp_path: Path) -> None:
    """Test that loading from non-existent path raises FileNotFoundError."""
    fake_path = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError, match="Metadata file not found"):
        mock_imputer._load_metadata(fake_path)


def test_save_load_roundtrip(
    mock_imputer: MockImputer,
    sample_dataloader: DataLoader,
    tmp_path: Path,
) -> None:
    """Test saving and loading preserves state."""
    # Fit the model
    mock_imputer.fit(sample_dataloader)
    original_name = mock_imputer.name
    original_params = mock_imputer.get_hyperparameters()

    # Save
    save_path = tmp_path / "model"
    mock_imputer.save(save_path)

    # Create new imputer and load
    new_imputer = MockImputer()
    assert not new_imputer.is_fitted  # Initially not fitted

    new_imputer.load(save_path)

    # Check state restored
    assert new_imputer.name == original_name
    assert new_imputer.is_fitted
    assert new_imputer.get_hyperparameters() == original_params


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_workflow(
    mock_imputer: MockImputer,
    sample_dataloader: DataLoader,
    sample_data: tuple[Tensor, Tensor],
    tmp_path: Path,
) -> None:
    """Test complete workflow: fit -> impute -> save -> load -> impute."""
    observed, mask = sample_data

    # Step 1: Fit
    mock_imputer.fit(sample_dataloader)
    assert mock_imputer.fit_called
    assert mock_imputer.is_fitted

    # Step 2: Impute
    result1 = mock_imputer.impute(observed, mask)
    assert mock_imputer.impute_called
    assert result1.shape == observed.shape

    # Step 3: Save
    save_path = tmp_path / "model"
    mock_imputer.save(save_path)
    assert mock_imputer.save_called

    # Step 4: Load into new instance
    new_imputer = MockImputer()
    new_imputer.load(save_path)
    assert new_imputer.load_called
    assert new_imputer.is_fitted

    # Step 5: Impute with loaded model
    result2 = new_imputer.impute(observed, mask)
    assert result2.shape == observed.shape


def test_multiple_impute_calls(
    mock_imputer: MockImputer,
    sample_dataloader: DataLoader,
    sample_data: tuple[Tensor, Tensor],
) -> None:
    """Test that impute() can be called multiple times after fit()."""
    observed, mask = sample_data
    mock_imputer.fit(sample_dataloader)

    # Call impute multiple times
    result1 = mock_imputer.impute(observed, mask)
    result2 = mock_imputer.impute(observed, mask)

    assert torch.equal(result1, result2)


def test_different_batch_sizes(
    mock_imputer: MockImputer, sample_dataloader: DataLoader
) -> None:
    """Test that impute() works with different batch sizes."""
    mock_imputer.fit(sample_dataloader)

    # Test with different batch sizes
    for batch_size in [1, 2, 4]:
        observed = torch.randn(batch_size, 10, 3)
        mask = torch.rand(batch_size, 10, 3) > 0.5
        result = mock_imputer.impute(observed, mask)
        assert result.shape == (batch_size, 10, 3)


def test_different_sequence_lengths(
    mock_imputer: MockImputer, sample_dataloader: DataLoader
) -> None:
    """Test that impute() works with different sequence lengths."""
    mock_imputer.fit(sample_dataloader)

    # Test with different sequence lengths
    for seq_len in [5, 10, 20, 50]:
        observed = torch.randn(2, seq_len, 3)
        mask = torch.rand(2, seq_len, 3) > 0.5
        result = mock_imputer.impute(observed, mask)
        assert result.shape == (2, seq_len, 3)


def test_different_variable_counts(
    mock_imputer: MockImputer, sample_dataloader: DataLoader
) -> None:
    """Test that impute() works with different variable counts."""
    mock_imputer.fit(sample_dataloader)

    # Test with different variable counts
    for n_vars in [1, 3, 6, 10]:
        observed = torch.randn(2, 10, n_vars)
        mask = torch.rand(2, 10, n_vars) > 0.5
        result = mock_imputer.impute(observed, mask)
        assert result.shape == (2, 10, n_vars)
