"""Tests for configuration classes."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from weather_imputation.config.base import BaseConfig, ExperimentConfig
from weather_imputation.config.data import (
    DataConfig,
    MaskingConfig,
    NormalizationConfig,
    SplitConfig,
    StationFilterConfig,
)


class SampleConfig(BaseConfig):
    """Test configuration class for testing base functionality."""

    name: str
    value: int = 42
    enabled: bool = True


def test_base_config():
    """Test basic BaseConfig functionality."""
    # Test instantiation
    config = SampleConfig(name="test")
    assert config.name == "test"
    assert config.value == 42
    assert config.enabled is True

    # Test with custom values
    config2 = SampleConfig(name="custom", value=100, enabled=False)
    assert config2.name == "custom"
    assert config2.value == 100
    assert config2.enabled is False


def test_base_config_validation():
    """Test that BaseConfig validates input."""
    # Missing required field should raise ValidationError
    with pytest.raises(ValidationError):
        SampleConfig()  # Missing 'name'

    # Invalid type should raise ValidationError
    with pytest.raises(ValidationError):
        SampleConfig(name="test", value="not_an_int")

    # Extra fields should raise ValidationError (extra='forbid')
    with pytest.raises(ValidationError):
        SampleConfig(name="test", extra_field="not_allowed")


def test_base_config_to_dict():
    """Test converting config to dictionary."""
    config = SampleConfig(name="test", value=99)
    result = config.to_dict()

    assert isinstance(result, dict)
    assert result["name"] == "test"
    assert result["value"] == 99
    assert result["enabled"] is True


def test_base_config_to_yaml():
    """Test converting config to YAML string."""
    config = SampleConfig(name="test", value=99)
    yaml_str = config.to_yaml()

    assert isinstance(yaml_str, str)
    assert "name: test" in yaml_str
    assert "value: 99" in yaml_str
    assert "enabled: true" in yaml_str


def test_base_config_from_dict():
    """Test creating config from dictionary."""
    data = {"name": "from_dict", "value": 123, "enabled": False}
    config = SampleConfig.from_dict(data)

    assert config.name == "from_dict"
    assert config.value == 123
    assert config.enabled is False


def test_base_config_from_yaml():
    """Test creating config from YAML string."""
    yaml_str = """
name: from_yaml
value: 456
enabled: false
"""
    config = SampleConfig.from_yaml(yaml_str)

    assert config.name == "from_yaml"
    assert config.value == 456
    assert config.enabled is False


def test_base_config_yaml_file_roundtrip():
    """Test saving to and loading from YAML file."""
    original = SampleConfig(name="roundtrip", value=789, enabled=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "config.yaml"

        # Save to file
        original.to_yaml_file(filepath)
        assert filepath.exists()

        # Load from file
        loaded = SampleConfig.from_yaml_file(filepath)
        assert loaded.name == original.name
        assert loaded.value == original.value
        assert loaded.enabled == original.enabled


def test_base_config_json_file_roundtrip():
    """Test saving to and loading from JSON file."""
    original = SampleConfig(name="json_test", value=999, enabled=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "config.json"

        # Save to file
        original.to_json_file(filepath)
        assert filepath.exists()

        # Load from file
        loaded = SampleConfig.from_json_file(filepath)
        assert loaded.name == original.name
        assert loaded.value == original.value
        assert loaded.enabled == original.enabled


def test_base_config_creates_parent_directories():
    """Test that file saving creates parent directories."""
    config = SampleConfig(name="test")

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "nested" / "dir" / "config.yaml"

        # Should create parent directories automatically
        config.to_yaml_file(filepath)
        assert filepath.exists()


def test_experiment_config():
    """Test ExperimentConfig with metadata fields."""
    config = ExperimentConfig(
        name="test_experiment",
        description="A test experiment",
        seed=123,
        tags=["baseline", "test"],
    )

    assert config.name == "test_experiment"
    assert config.description == "A test experiment"
    assert config.seed == 123
    assert config.tags == ["baseline", "test"]


def test_experiment_config_defaults():
    """Test ExperimentConfig default values."""
    config = ExperimentConfig(name="minimal")

    assert config.name == "minimal"
    assert config.description == ""
    assert config.seed == 42
    assert config.tags == []


def test_experiment_config_str():
    """Test ExperimentConfig string representation."""
    config = ExperimentConfig(name="my_experiment")
    str_repr = str(config)

    assert "ExperimentConfig" in str_repr
    assert "my_experiment" in str_repr


def test_validation_on_assignment():
    """Test that BaseConfig validates on assignment."""
    config = SampleConfig(name="test")

    # Valid assignment should work
    config.value = 200
    assert config.value == 200

    # Invalid assignment should raise ValidationError
    with pytest.raises(ValidationError):
        config.value = "not_an_int"


def test_nested_config():
    """Test BaseConfig with nested configuration."""

    class NestedConfig(BaseConfig):
        inner: SampleConfig
        multiplier: int = 2

    nested = NestedConfig(inner=SampleConfig(name="inner_test", value=10))
    assert nested.inner.name == "inner_test"
    assert nested.inner.value == 10
    assert nested.multiplier == 2

    # Test serialization with nested config
    yaml_str = nested.to_yaml()
    assert "inner:" in yaml_str
    assert "name: inner_test" in yaml_str

    # Test deserialization with nested config
    loaded = NestedConfig.from_yaml(yaml_str)
    assert loaded.inner.name == "inner_test"
    assert loaded.inner.value == 10


# ================================
# Data Configuration Tests
# ================================


def test_station_filter_config():
    """Test StationFilterConfig with default values."""
    config = StationFilterConfig()

    # Check default completeness thresholds
    assert config.min_temperature_completeness == 60.0
    assert config.min_dew_point_completeness == 60.0
    assert config.min_sea_level_pressure_completeness == 60.0
    assert config.min_wind_speed_completeness == 60.0
    assert config.min_wind_direction_completeness == 60.0
    assert config.min_relative_humidity_completeness == 60.0

    # Check default temporal thresholds
    assert config.min_years_available == 3
    assert config.min_total_observations == 1000

    # Check default geographic bounds (North America)
    assert config.latitude_range == (24.0, 50.0)
    assert config.longitude_range == (-130.0, -60.0)

    # Check gap pattern threshold
    assert config.max_gap_duration_hours is None


def test_station_filter_config_custom():
    """Test StationFilterConfig with custom values."""
    config = StationFilterConfig(
        min_temperature_completeness=80.0,
        min_years_available=5,
        latitude_range=(30.0, 45.0),
        longitude_range=(-120.0, -70.0),
        max_gap_duration_hours=720.0,
    )

    assert config.min_temperature_completeness == 80.0
    assert config.min_years_available == 5
    assert config.latitude_range == (30.0, 45.0)
    assert config.longitude_range == (-120.0, -70.0)
    assert config.max_gap_duration_hours == 720.0


def test_station_filter_config_validation():
    """Test StationFilterConfig validation."""
    # Invalid completeness (> 100)
    with pytest.raises(ValidationError):
        StationFilterConfig(min_temperature_completeness=150.0)

    # Invalid completeness (< 0)
    with pytest.raises(ValidationError):
        StationFilterConfig(min_temperature_completeness=-10.0)

    # Invalid years (< 1)
    with pytest.raises(ValidationError):
        StationFilterConfig(min_years_available=0)

    # Invalid latitude range (min >= max)
    with pytest.raises(ValidationError):
        StationFilterConfig(latitude_range=(50.0, 24.0))

    # Invalid longitude range (min >= max)
    with pytest.raises(ValidationError):
        StationFilterConfig(longitude_range=(-60.0, -130.0))


def test_normalization_config():
    """Test NormalizationConfig with default values."""
    config = NormalizationConfig()

    assert config.method == "zscore"
    assert config.per_station is True
    assert config.clip_outliers is False


def test_normalization_config_custom():
    """Test NormalizationConfig with custom values."""
    config = NormalizationConfig(
        method="minmax", per_station=False, clip_outliers=True
    )

    assert config.method == "minmax"
    assert config.per_station is False
    assert config.clip_outliers is True


def test_normalization_config_validation():
    """Test NormalizationConfig validation."""
    # Invalid method
    with pytest.raises(ValidationError):
        NormalizationConfig(method="invalid_method")


def test_masking_config():
    """Test MaskingConfig with default values."""
    config = MaskingConfig()

    assert config.strategy == "realistic"
    assert config.missing_ratio == 0.2
    assert config.min_gap_length == 1
    assert config.max_gap_length == 168


def test_masking_config_custom():
    """Test MaskingConfig with custom values."""
    config = MaskingConfig(
        strategy="mcar", missing_ratio=0.3, min_gap_length=6, max_gap_length=72
    )

    assert config.strategy == "mcar"
    assert config.missing_ratio == 0.3
    assert config.min_gap_length == 6
    assert config.max_gap_length == 72


def test_masking_config_validation():
    """Test MaskingConfig validation."""
    # Invalid strategy
    with pytest.raises(ValidationError):
        MaskingConfig(strategy="invalid_strategy")

    # Invalid missing_ratio (> 1)
    with pytest.raises(ValidationError):
        MaskingConfig(missing_ratio=1.5)

    # Invalid missing_ratio (< 0)
    with pytest.raises(ValidationError):
        MaskingConfig(missing_ratio=-0.1)

    # Invalid gap length (max < min)
    with pytest.raises(ValidationError):
        MaskingConfig(min_gap_length=10, max_gap_length=5)


def test_split_config():
    """Test SplitConfig with default values."""
    config = SplitConfig()

    assert config.strategy == "simulated"
    assert config.train_ratio == 0.7
    assert config.val_ratio == 0.15
    assert config.test_ratio == 0.15


def test_split_config_custom():
    """Test SplitConfig with custom values."""
    config = SplitConfig(
        strategy="temporal", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )

    assert config.strategy == "temporal"
    assert config.train_ratio == 0.6
    assert config.val_ratio == 0.2
    assert config.test_ratio == 0.2


def test_split_config_validation():
    """Test SplitConfig validation."""
    # Invalid strategy
    with pytest.raises(ValidationError):
        SplitConfig(strategy="invalid_strategy")

    # Invalid ratio (> 1)
    with pytest.raises(ValidationError):
        SplitConfig(train_ratio=1.5)

    # Invalid ratio (< 0)
    with pytest.raises(ValidationError):
        SplitConfig(val_ratio=-0.1)

    # Ratios don't sum to 1.0
    with pytest.raises(ValidationError):
        SplitConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_data_config():
    """Test DataConfig with default values."""
    config = DataConfig()

    # Check Tier 1 variables (6 core variables)
    assert len(config.variables) == 6
    assert "temperature" in config.variables
    assert "dew_point_temperature" in config.variables
    assert "sea_level_pressure" in config.variables
    assert "wind_speed" in config.variables
    assert "wind_direction" in config.variables
    assert "relative_humidity" in config.variables

    # Check sub-configurations
    assert isinstance(config.station_filter, StationFilterConfig)
    assert isinstance(config.normalization, NormalizationConfig)
    assert isinstance(config.masking, MaskingConfig)
    assert isinstance(config.split, SplitConfig)

    # Check windowing parameters
    assert config.window_size == 168
    assert config.stride == 24

    # Check quality control
    assert config.report_types == ["AUTO", "FM-15"]


def test_data_config_custom():
    """Test DataConfig with custom values."""
    config = DataConfig(
        variables=["temperature", "dew_point_temperature"],
        station_filter=StationFilterConfig(min_temperature_completeness=80.0),
        normalization=NormalizationConfig(method="minmax"),
        masking=MaskingConfig(strategy="mcar", missing_ratio=0.3),
        split=SplitConfig(strategy="temporal"),
        window_size=336,
        stride=48,
        report_types=["AUTO"],
    )

    assert len(config.variables) == 2
    assert config.station_filter.min_temperature_completeness == 80.0
    assert config.normalization.method == "minmax"
    assert config.masking.strategy == "mcar"
    assert config.split.strategy == "temporal"
    assert config.window_size == 336
    assert config.stride == 48
    assert config.report_types == ["AUTO"]


def test_data_config_validation():
    """Test DataConfig validation."""
    # Invalid variable (not in Tier 1)
    with pytest.raises(ValidationError):
        DataConfig(variables=["temperature", "invalid_variable"])

    # Empty variables list
    with pytest.raises(ValidationError):
        DataConfig(variables=[])

    # Invalid window_size (< 1)
    with pytest.raises(ValidationError):
        DataConfig(window_size=0)

    # Invalid stride (< 1)
    with pytest.raises(ValidationError):
        DataConfig(stride=0)

    # Empty report_types list
    with pytest.raises(ValidationError):
        DataConfig(report_types=[])


def test_data_config_yaml_roundtrip():
    """Test DataConfig serialization and deserialization."""
    original = DataConfig(
        variables=["temperature", "wind_speed"],
        station_filter=StationFilterConfig(min_temperature_completeness=75.0),
        window_size=240,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "data_config.yaml"

        # Save to YAML file
        original.to_yaml_file(filepath)
        assert filepath.exists()

        # Load from YAML file
        loaded = DataConfig.from_yaml_file(filepath)

        # Verify all fields match
        assert loaded.variables == original.variables
        assert (
            loaded.station_filter.min_temperature_completeness
            == original.station_filter.min_temperature_completeness
        )
        assert loaded.window_size == original.window_size


def test_data_config_nested_serialization():
    """Test that DataConfig correctly serializes nested configurations."""
    config = DataConfig(
        variables=["temperature", "dew_point_temperature"],
        station_filter=StationFilterConfig(
            min_temperature_completeness=70.0, min_years_available=5
        ),
        normalization=NormalizationConfig(method="minmax", per_station=False),
    )

    yaml_str = config.to_yaml()

    # Check that nested configs are properly serialized
    assert "station_filter:" in yaml_str
    assert "min_temperature_completeness: 70.0" in yaml_str
    assert "min_years_available: 5" in yaml_str
    assert "normalization:" in yaml_str
    assert "method: minmax" in yaml_str
    assert "per_station: false" in yaml_str


# ============================================================================
# Model Configuration Tests
# ============================================================================


def test_linear_interpolation_config():
    """Test LinearInterpolationConfig with default values."""
    from weather_imputation.config.model import LinearInterpolationConfig

    config = LinearInterpolationConfig()

    assert config.model_type == "linear"
    assert config.n_variables == 6
    assert config.max_gap_size is None


def test_linear_interpolation_config_custom():
    """Test LinearInterpolationConfig with custom values."""
    from weather_imputation.config.model import LinearInterpolationConfig

    config = LinearInterpolationConfig(n_variables=4, max_gap_size=24)

    assert config.model_type == "linear"
    assert config.n_variables == 4
    assert config.max_gap_size == 24


def test_spline_interpolation_config():
    """Test SplineInterpolationConfig with default values."""
    from weather_imputation.config.model import SplineInterpolationConfig

    config = SplineInterpolationConfig()

    assert config.model_type == "spline"
    assert config.n_variables == 6
    assert config.spline_order == 3
    assert config.max_gap_size is None


def test_spline_interpolation_config_custom():
    """Test SplineInterpolationConfig with custom values."""
    from weather_imputation.config.model import SplineInterpolationConfig

    config = SplineInterpolationConfig(spline_order=5, max_gap_size=48)

    assert config.model_type == "spline"
    assert config.spline_order == 5
    assert config.max_gap_size == 48


def test_spline_interpolation_config_validation():
    """Test SplineInterpolationConfig validation."""
    from weather_imputation.config.model import SplineInterpolationConfig

    # Invalid spline order (< 1)
    with pytest.raises(ValidationError):
        SplineInterpolationConfig(spline_order=0)

    # Invalid spline order (> 5)
    with pytest.raises(ValidationError):
        SplineInterpolationConfig(spline_order=6)


def test_mice_config():
    """Test MICEConfig with default values."""
    from weather_imputation.config.model import MICEConfig

    config = MICEConfig()

    assert config.model_type == "mice"
    assert config.n_variables == 6
    assert config.n_iterations == 10
    assert config.n_imputations == 5
    assert config.predictor_method == "bayesian_ridge"


def test_mice_config_custom():
    """Test MICEConfig with custom values."""
    from weather_imputation.config.model import MICEConfig

    config = MICEConfig(
        n_iterations=20, n_imputations=10, predictor_method="random_forest"
    )

    assert config.n_iterations == 20
    assert config.n_imputations == 10
    assert config.predictor_method == "random_forest"


def test_saits_config():
    """Test SAITSConfig with default values."""
    from weather_imputation.config.model import SAITSConfig

    config = SAITSConfig()

    assert config.model_type == "saits"
    assert config.n_variables == 6
    assert config.n_layers == 2
    assert config.n_heads == 4
    assert config.d_model == 128
    assert config.d_ff == 512
    assert config.dropout == 0.1
    assert config.mit_weight == 1.0
    assert config.ort_weight == 1.0
    assert config.use_learnable_position_encoding is True
    assert config.max_seq_len == 512
    assert config.circular_wind_encoding is True
    assert config.use_station_metadata is False


def test_saits_config_custom():
    """Test SAITSConfig with custom values."""
    from weather_imputation.config.model import SAITSConfig

    config = SAITSConfig(
        n_layers=4,
        n_heads=8,
        d_model=256,
        d_ff=1024,
        dropout=0.2,
        mit_weight=1.5,
        ort_weight=0.5,
        use_station_metadata=True,
    )

    assert config.n_layers == 4
    assert config.n_heads == 8
    assert config.d_model == 256
    assert config.d_ff == 1024
    assert config.dropout == 0.2
    assert config.mit_weight == 1.5
    assert config.ort_weight == 0.5
    assert config.use_station_metadata is True


def test_saits_config_validation():
    """Test SAITSConfig validation."""
    from weather_imputation.config.model import SAITSConfig

    # Valid: d_model divisible by n_heads
    config = SAITSConfig(d_model=128, n_heads=4)
    assert config.d_model == 128

    # Invalid: d_model not divisible by n_heads (101 % 4 = 1, not divisible)
    with pytest.raises(ValidationError) as exc_info:
        SAITSConfig(d_model=101, n_heads=4)
    assert "divisible by n_heads" in str(exc_info.value)


def test_csdi_config():
    """Test CSDIConfig with default values."""
    from weather_imputation.config.model import CSDIConfig

    config = CSDIConfig()

    assert config.model_type == "csdi"
    assert config.n_variables == 6
    assert config.n_diffusion_steps == 50
    assert config.noise_schedule == "cosine"
    assert config.beta_start == 0.0001
    assert config.beta_end == 0.02
    assert config.n_layers == 4
    assert config.n_heads == 4
    assert config.d_model == 128
    assert config.d_ff == 512
    assert config.dropout == 0.1
    assert config.n_samples == 10
    assert config.sampling_strategy == "ddpm"
    assert config.ddim_steps is None
    assert config.use_time_embedding is True
    assert config.use_station_metadata is False


def test_csdi_config_custom():
    """Test CSDIConfig with custom values."""
    from weather_imputation.config.model import CSDIConfig

    config = CSDIConfig(
        n_diffusion_steps=100,
        noise_schedule="linear",
        beta_start=0.0005,
        beta_end=0.05,
        n_layers=6,
        n_heads=8,
        d_model=256,
        n_samples=20,
        sampling_strategy="ddim",
        ddim_steps=25,
        use_station_metadata=True,
    )

    assert config.n_diffusion_steps == 100
    assert config.noise_schedule == "linear"
    assert config.beta_start == 0.0005
    assert config.beta_end == 0.05
    assert config.n_layers == 6
    assert config.n_heads == 8
    assert config.d_model == 256
    assert config.n_samples == 20
    assert config.sampling_strategy == "ddim"
    assert config.ddim_steps == 25
    assert config.use_station_metadata is True


def test_csdi_config_validation():
    """Test CSDIConfig validation."""
    from weather_imputation.config.model import CSDIConfig

    # Valid: d_model divisible by n_heads
    config = CSDIConfig(d_model=128, n_heads=4)
    assert config.d_model == 128

    # Invalid: d_model not divisible by n_heads (101 % 4 = 1, not divisible)
    with pytest.raises(ValidationError) as exc_info:
        CSDIConfig(d_model=101, n_heads=4)
    assert "divisible by n_heads" in str(exc_info.value)

    # Valid: beta_end > beta_start
    config = CSDIConfig(beta_start=0.0001, beta_end=0.02)
    assert config.beta_end > config.beta_start

    # Invalid: beta_end <= beta_start
    with pytest.raises(ValidationError) as exc_info:
        CSDIConfig(beta_start=0.02, beta_end=0.01)
    assert "greater than beta_start" in str(exc_info.value)


def test_model_config_yaml_roundtrip():
    """Test model configurations YAML serialization and deserialization."""
    from weather_imputation.config.model import SAITSConfig

    original = SAITSConfig(
        n_layers=4,
        n_heads=8,
        d_model=256,
        mit_weight=1.5,
        ort_weight=0.5,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "saits_config.yaml"

        # Save to YAML file
        original.to_yaml_file(filepath)
        assert filepath.exists()

        # Load from YAML file
        loaded = SAITSConfig.from_yaml_file(filepath)

        # Verify all custom fields match
        assert loaded.n_layers == original.n_layers
        assert loaded.n_heads == original.n_heads
        assert loaded.d_model == original.d_model
        assert loaded.mit_weight == original.mit_weight
        assert loaded.ort_weight == original.ort_weight


# ============================================================================
# Training Configuration Tests
# ============================================================================


def test_optimizer_config():
    """Test OptimizerConfig with default values."""
    from weather_imputation.config.training import OptimizerConfig

    config = OptimizerConfig()

    assert config.optimizer_type == "adamw"
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 1e-4
    assert config.betas == (0.9, 0.999)
    assert config.momentum == 0.9
    assert config.grad_clip_norm == 1.0
    assert config.grad_clip_value is None


def test_optimizer_config_custom():
    """Test OptimizerConfig with custom values."""
    from weather_imputation.config.training import OptimizerConfig

    config = OptimizerConfig(
        optimizer_type="adam",
        learning_rate=5e-4,
        weight_decay=1e-5,
        betas=(0.95, 0.999),
        grad_clip_norm=5.0,
        grad_clip_value=1.0,
    )

    assert config.optimizer_type == "adam"
    assert config.learning_rate == 5e-4
    assert config.weight_decay == 1e-5
    assert config.betas == (0.95, 0.999)
    assert config.grad_clip_norm == 5.0
    assert config.grad_clip_value == 1.0


def test_optimizer_config_validation():
    """Test OptimizerConfig validation."""
    from weather_imputation.config.training import OptimizerConfig

    # Invalid learning rate (negative)
    with pytest.raises(ValidationError):
        OptimizerConfig(learning_rate=-0.001)

    # Invalid learning rate (> 1.0)
    with pytest.raises(ValidationError):
        OptimizerConfig(learning_rate=1.5)

    # Invalid betas (>= 1.0)
    with pytest.raises(ValidationError) as exc_info:
        OptimizerConfig(betas=(0.9, 1.0))
    assert "Beta coefficients" in str(exc_info.value)

    # Invalid betas (negative)
    with pytest.raises(ValidationError) as exc_info:
        OptimizerConfig(betas=(-0.1, 0.999))
    assert "Beta coefficients" in str(exc_info.value)


def test_scheduler_config():
    """Test SchedulerConfig with default values."""
    from weather_imputation.config.training import SchedulerConfig

    config = SchedulerConfig()

    assert config.scheduler_type == "cosine"
    assert config.warmup_epochs == 5
    assert config.patience == 5
    assert config.factor == 0.5
    assert config.step_size == 10
    assert config.gamma == 0.1
    assert config.min_lr == 1e-6


def test_scheduler_config_custom():
    """Test SchedulerConfig with custom values."""
    from weather_imputation.config.training import SchedulerConfig

    config = SchedulerConfig(
        scheduler_type="plateau",
        warmup_epochs=10,
        patience=10,
        factor=0.1,
        min_lr=1e-7,
    )

    assert config.scheduler_type == "plateau"
    assert config.warmup_epochs == 10
    assert config.patience == 10
    assert config.factor == 0.1
    assert config.min_lr == 1e-7


def test_scheduler_config_validation():
    """Test SchedulerConfig validation."""
    from weather_imputation.config.training import SchedulerConfig

    # Invalid factor (>= 1.0)
    with pytest.raises(ValidationError):
        SchedulerConfig(factor=1.0)

    # Invalid gamma (> 1.0)
    with pytest.raises(ValidationError):
        SchedulerConfig(gamma=1.5)

    # Invalid patience (< 1)
    with pytest.raises(ValidationError):
        SchedulerConfig(patience=0)


def test_early_stopping_config():
    """Test EarlyStoppingConfig with default values."""
    from weather_imputation.config.training import EarlyStoppingConfig

    config = EarlyStoppingConfig()

    assert config.enabled is True
    assert config.patience == 10
    assert config.min_delta == 1e-4
    assert config.monitor == "val_loss"
    assert config.mode == "min"


def test_early_stopping_config_custom():
    """Test EarlyStoppingConfig with custom values."""
    from weather_imputation.config.training import EarlyStoppingConfig

    config = EarlyStoppingConfig(
        enabled=False, patience=15, min_delta=1e-3, monitor="val_rmse", mode="min"
    )

    assert config.enabled is False
    assert config.patience == 15
    assert config.min_delta == 1e-3
    assert config.monitor == "val_rmse"
    assert config.mode == "min"


def test_early_stopping_config_validation():
    """Test EarlyStoppingConfig validation."""
    from weather_imputation.config.training import EarlyStoppingConfig

    # Invalid patience (< 1)
    with pytest.raises(ValidationError):
        EarlyStoppingConfig(patience=0)

    # Invalid min_delta (negative)
    with pytest.raises(ValidationError):
        EarlyStoppingConfig(min_delta=-0.001)


def test_checkpoint_config():
    """Test CheckpointConfig with default values."""
    from weather_imputation.config.training import CheckpointConfig

    config = CheckpointConfig()

    assert config.save_every_n_epochs == 1
    assert config.save_every_n_minutes == 30
    assert config.keep_last_n == 3
    assert config.save_best is True
    assert config.monitor == "val_loss"
    assert config.mode == "min"


def test_checkpoint_config_custom():
    """Test CheckpointConfig with custom values."""
    from weather_imputation.config.training import CheckpointConfig

    config = CheckpointConfig(
        save_every_n_epochs=5,
        save_every_n_minutes=60,
        keep_last_n=5,
        save_best=True,
        monitor="val_rmse",
        mode="min",
    )

    assert config.save_every_n_epochs == 5
    assert config.save_every_n_minutes == 60
    assert config.keep_last_n == 5
    assert config.save_best is True
    assert config.monitor == "val_rmse"


def test_checkpoint_config_validation():
    """Test CheckpointConfig validation."""
    from weather_imputation.config.training import CheckpointConfig

    # Valid: at least one method enabled
    config = CheckpointConfig(save_every_n_epochs=1)
    assert config.save_every_n_epochs == 1

    config = CheckpointConfig(save_every_n_epochs=0, save_every_n_minutes=30)
    assert config.save_every_n_minutes == 30

    config = CheckpointConfig(
        save_every_n_epochs=0, save_every_n_minutes=0, save_best=True
    )
    assert config.save_best is True

    # Invalid: all checkpoint methods disabled
    with pytest.raises(ValidationError) as exc_info:
        CheckpointConfig(save_every_n_epochs=0, save_every_n_minutes=0, save_best=False)
    assert "at least one checkpoint method" in str(exc_info.value).lower()


def test_training_config():
    """Test TrainingConfig with default values."""
    from weather_imputation.config.training import TrainingConfig

    config = TrainingConfig()

    assert config.batch_size == 32
    assert config.max_epochs == 100
    assert config.validation_frequency is None
    assert config.seed == 42
    assert config.device == "auto"
    assert config.mixed_precision is True
    assert config.compile_model is False
    assert config.num_workers == 4

    # Check nested configs are initialized
    assert config.optimizer.optimizer_type == "adamw"
    assert config.scheduler.scheduler_type == "cosine"
    assert config.early_stopping.enabled is True
    assert config.checkpoint.save_best is True


def test_training_config_custom():
    """Test TrainingConfig with custom values."""
    from weather_imputation.config.training import (
        CheckpointConfig,
        EarlyStoppingConfig,
        OptimizerConfig,
        SchedulerConfig,
        TrainingConfig,
    )

    config = TrainingConfig(
        batch_size=64,
        max_epochs=200,
        validation_frequency=100,
        seed=123,
        device="cuda",
        mixed_precision=False,
        compile_model=True,
        num_workers=8,
        optimizer=OptimizerConfig(learning_rate=1e-4, optimizer_type="adam"),
        scheduler=SchedulerConfig(scheduler_type="plateau", warmup_epochs=0),
        early_stopping=EarlyStoppingConfig(enabled=False),
        checkpoint=CheckpointConfig(save_every_n_epochs=5, keep_last_n=10),
    )

    assert config.batch_size == 64
    assert config.max_epochs == 200
    assert config.validation_frequency == 100
    assert config.seed == 123
    assert config.device == "cuda"
    assert config.mixed_precision is False
    assert config.compile_model is True
    assert config.num_workers == 8

    # Check nested configs
    assert config.optimizer.learning_rate == 1e-4
    assert config.optimizer.optimizer_type == "adam"
    assert config.scheduler.scheduler_type == "plateau"
    assert config.scheduler.warmup_epochs == 0
    assert config.early_stopping.enabled is False
    assert config.checkpoint.save_every_n_epochs == 5
    assert config.checkpoint.keep_last_n == 10


def test_training_config_validation():
    """Test TrainingConfig validation."""
    from weather_imputation.config.training import TrainingConfig

    # Invalid batch size (< 1)
    with pytest.raises(ValidationError):
        TrainingConfig(batch_size=0)

    # Invalid max_epochs (< 1)
    with pytest.raises(ValidationError):
        TrainingConfig(max_epochs=0)

    # Invalid num_workers (negative)
    with pytest.raises(ValidationError):
        TrainingConfig(num_workers=-1)


def test_training_config_yaml_roundtrip():
    """Test TrainingConfig YAML serialization and deserialization."""
    from weather_imputation.config.training import (
        OptimizerConfig,
        SchedulerConfig,
        TrainingConfig,
    )

    original = TrainingConfig(
        batch_size=64,
        max_epochs=50,
        seed=999,
        optimizer=OptimizerConfig(learning_rate=5e-4, weight_decay=1e-5),
        scheduler=SchedulerConfig(scheduler_type="step", step_size=20),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "training_config.yaml"

        # Save to YAML file
        original.to_yaml_file(filepath)
        assert filepath.exists()

        # Load from YAML file
        loaded = TrainingConfig.from_yaml_file(filepath)

        # Verify all custom fields match
        assert loaded.batch_size == original.batch_size
        assert loaded.max_epochs == original.max_epochs
        assert loaded.seed == original.seed
        assert loaded.optimizer.learning_rate == original.optimizer.learning_rate
        assert loaded.optimizer.weight_decay == original.optimizer.weight_decay
        assert loaded.scheduler.scheduler_type == original.scheduler.scheduler_type
        assert loaded.scheduler.step_size == original.scheduler.step_size


def test_training_config_nested_serialization():
    """Test TrainingConfig nested configuration serialization."""
    from weather_imputation.config.training import TrainingConfig

    config = TrainingConfig(batch_size=128, max_epochs=75)

    # Convert to dict
    config_dict = config.model_dump()
    assert config_dict["batch_size"] == 128
    assert config_dict["max_epochs"] == 75
    assert "optimizer" in config_dict
    assert "scheduler" in config_dict
    assert "early_stopping" in config_dict
    assert "checkpoint" in config_dict

    # Verify nested configs are dicts
    assert isinstance(config_dict["optimizer"], dict)
    assert config_dict["optimizer"]["optimizer_type"] == "adamw"
    assert isinstance(config_dict["scheduler"], dict)
    assert config_dict["scheduler"]["scheduler_type"] == "cosine"
