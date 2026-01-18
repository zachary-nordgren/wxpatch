"""System resource utilities."""

import logging
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SystemResources:
    """Container for system resource information."""

    system: str
    processor: str
    python_version: str
    total_memory_gb: float
    available_memory_gb: float
    free_disk_space_gb: float | None
    memory_warning: bool
    disk_warning: bool


def check_system_resources(check_path: Path | None = None) -> SystemResources:
    """Check available system resources and provide warnings.

    Args:
        check_path: Path to check for disk space (defaults to current directory)

    Returns:
        SystemResources dataclass with resource information
    """
    # Try to import psutil, but make it optional
    try:
        import psutil

        mem = psutil.virtual_memory()
        total_memory_gb = mem.total / (1024**3)
        available_memory_gb = mem.available / (1024**3)
    except ImportError:
        logger.info("psutil not installed, memory info unavailable")
        total_memory_gb = 0.0
        available_memory_gb = 0.0

    # Memory warning threshold
    memory_warning = available_memory_gb < 2.0 and available_memory_gb > 0

    if memory_warning:
        logger.warning("Low memory detected! Consider reducing concurrent operations.")

    # Check disk space
    disk_path = check_path or Path.cwd()
    free_disk_space_gb: float | None = None
    disk_warning = False

    try:
        disk_usage = shutil.disk_usage(disk_path)
        free_disk_space_gb = disk_usage.free / (1024**3)

        if free_disk_space_gb < 10:
            logger.warning(
                f"Low disk space ({free_disk_space_gb:.1f}GB free). "
                "You may not have enough space for all data."
            )
            disk_warning = True
    except OSError as e:
        logger.info(f"Could not check disk space: {e}")

    return SystemResources(
        system=platform.system(),
        processor=platform.processor(),
        python_version=platform.python_version(),
        total_memory_gb=total_memory_gb,
        available_memory_gb=available_memory_gb,
        free_disk_space_gb=free_disk_space_gb,
        memory_warning=memory_warning,
        disk_warning=disk_warning,
    )
