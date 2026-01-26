"""Base configuration classes for weather imputation.

This module provides base Pydantic models that other configuration classes inherit from,
offering common functionality for validation, serialization, and YAML/JSON loading.
"""

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel):
    """Base configuration class with common functionality.

    Provides:
    - Pydantic v2 configuration
    - YAML/JSON serialization
    - File loading utilities
    - Validation helpers

    Example:
        >>> class MyConfig(BaseConfig):
        ...     name: str
        ...     value: int = 42
        >>> config = MyConfig(name="test")
        >>> config.to_yaml_file("config.yaml")
        >>> loaded = MyConfig.from_yaml_file("config.yaml")
    """

    model_config = ConfigDict(
        # Allow arbitrary types (for Path, etc.)
        arbitrary_types_allowed=True,
        # Validate on assignment
        validate_assignment=True,
        # Use enum values instead of enum objects
        use_enum_values=True,
        # Extra fields raise validation error
        extra="forbid",
        # Populate by name (allows field aliases)
        populate_by_name=True,
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation with all fields
        """
        return self.model_dump(mode="python")

    def to_yaml(self) -> str:
        """Convert configuration to YAML string.

        Returns:
            YAML string representation
        """
        return yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_yaml_file(self, path: Path | str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_yaml())

    def to_json_file(self, path: Path | str) -> None:
        """Save configuration to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            Validated configuration instance
        """
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls: type[T], yaml_str: str) -> T:
        """Create configuration from YAML string.

        Args:
            yaml_str: YAML string

        Returns:
            Validated configuration instance
        """
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    @classmethod
    def from_yaml_file(cls: type[T], path: Path | str) -> T:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Validated configuration instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValidationError: If configuration is invalid
        """
        path = Path(path)
        with open(path) as f:
            return cls.from_yaml(f.read())

    @classmethod
    def from_json_file(cls: type[T], path: Path | str) -> T:
        """Load configuration from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Validated configuration instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValidationError: If configuration is invalid
        """
        path = Path(path)
        with open(path) as f:
            import json

            data = json.load(f)
        return cls.from_dict(data)


class ExperimentConfig(BaseConfig):
    """Base configuration for experiments with metadata tracking.

    Adds common fields for experiment tracking and reproducibility.
    """

    name: str = Field(..., description="Experiment name")
    description: str = Field(default="", description="Experiment description")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    tags: list[str] = Field(default_factory=list, description="Experiment tags for organization")

    def __str__(self) -> str:
        """String representation showing experiment name."""
        return f"{self.__class__.__name__}(name='{self.name}')"
