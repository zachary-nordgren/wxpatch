"""Tests for configuration classes."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import Field, ValidationError

from weather_imputation.config.base import BaseConfig, ExperimentConfig


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
