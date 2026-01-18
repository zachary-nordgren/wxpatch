"""Filesystem utilities for weather imputation."""

import logging
from pathlib import Path

from weather_imputation.config.paths import (
    GHCNH_RAW_DIR,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """Create all required data directories if they don't exist."""
    directories = [
        GHCNH_RAW_DIR,
        PROCESSED_DIR,
    ]

    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")


def list_year_directories() -> list[int]:
    """List all year directories in the GHCNh raw data folder.

    Returns:
        Sorted list of years that have data directories
    """
    if not GHCNH_RAW_DIR.exists():
        return []

    years = []
    for path in GHCNH_RAW_DIR.iterdir():
        if path.is_dir():
            try:
                year = int(path.name)
                years.append(year)
            except ValueError:
                continue

    return sorted(years)


def list_stations_for_year(year: int) -> list[str]:
    """List all station IDs that have data for a given year.

    Args:
        year: Year to check

    Returns:
        List of station IDs (without .parquet extension)
    """
    year_dir = GHCNH_RAW_DIR / str(year)
    if not year_dir.exists():
        return []

    stations = []
    for path in year_dir.iterdir():
        if path.is_file() and path.suffix == ".parquet":
            stations.append(path.stem)

    return sorted(stations)


def count_parquet_files() -> dict[int, int]:
    """Count parquet files per year.

    Returns:
        Dictionary mapping year to file count
    """
    counts: dict[int, int] = {}
    for year in list_year_directories():
        stations = list_stations_for_year(year)
        counts[year] = len(stations)
    return counts


def get_total_size_bytes(year: int | None = None) -> int:
    """Get total size of parquet files in bytes.

    Args:
        year: If specified, only count files for this year

    Returns:
        Total size in bytes
    """
    total = 0

    if year is not None:
        year_dir = GHCNH_RAW_DIR / str(year)
        if year_dir.exists():
            for path in year_dir.rglob("*.parquet"):
                total += path.stat().st_size
    else:
        for path in GHCNH_RAW_DIR.rglob("*.parquet"):
            total += path.stat().st_size

    return total


def format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.23 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0  # type: ignore
    return f"{size_bytes:.2f} PB"
