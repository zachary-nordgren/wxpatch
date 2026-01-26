"""Path configuration for weather imputation project."""

from pathlib import Path

# Get the project root (two levels up from this file:
# config -> weather_imputation -> src -> project)
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent.parent.parent

# Main data directories
DATA_DIR = PROJECT_ROOT / "data"
GHCNH_RAW_DIR = DATA_DIR / "raw" / "ghcnh"
PROCESSED_DIR = DATA_DIR / "processed"

# Metadata files
METADATA_PARQUET = PROCESSED_DIR / "metadata.parquet"
METADATA_CLEANED_PARQUET = PROCESSED_DIR / "metadata_cleaned.parquet"
METADATA_CSV = PROCESSED_DIR / "metadata.csv"
METADATA_CLEANED_CSV = PROCESSED_DIR / "metadata_cleaned.csv"
CLEANING_REPORT_JSON = PROCESSED_DIR / "cleaning_report.json"
CLEANING_LOG_TXT = PROCESSED_DIR / "clean_log.txt"

# Station list from NOAA
STATION_LIST_PATH = GHCNH_RAW_DIR / "ghcnh-station-list.csv"


def get_station_year_path(station_id: str, year: int) -> Path:
    """Get the path to a station's parquet file for a specific year.

    Args:
        station_id: Station identifier (e.g., "USW00003046")
        year: Year of the data file

    Returns:
        Path to the parquet file: data/raw/ghcnh/{year}/GHCNh_{station_id}_{year}.parquet
    """
    return GHCNH_RAW_DIR / str(year) / f"GHCNh_{station_id}_{year}.parquet"


def get_year_dir(year: int) -> Path:
    """Get the directory containing all station files for a year.

    Args:
        year: Year of the data

    Returns:
        Path to the year directory: data/raw/ghcnh/{year}/
    """
    return GHCNH_RAW_DIR / str(year)


def ensure_directories() -> None:
    """Create all required directories if they don't exist."""
    GHCNH_RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
