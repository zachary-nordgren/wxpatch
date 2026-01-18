"""Download configuration for GHCNh data."""

from pydantic import BaseModel, Field


class DownloadSettings(BaseModel):
    """Configuration for GHCNh data downloads."""

    # NOAA GHCNh parquet data URLs
    base_url: str = Field(
        default="https://www.ncei.noaa.gov/oa/global-hourly-data/ghcnh-parquet/",
        description="Base URL for GHCNh parquet files",
    )
    station_list_url: str = Field(
        default="https://www.ncei.noaa.gov/oa/global-hourly-data/ghcnh-station-list.csv",
        description="URL for station metadata list",
    )

    # Network settings
    timeout_seconds: int = Field(default=300, ge=10, le=3600)
    chunk_size: int = Field(default=8192, ge=1024)
    max_retries: int = Field(default=3, ge=1, le=10)
    max_concurrent_downloads: int = Field(default=8, ge=1, le=32)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)

    # Download behavior
    verify_ssl: bool = Field(default=True)
    user_agent: str = Field(
        default="weather-imputation/0.1.0 (research; contact: github.com/user/repo)"
    )


# Default settings instance
DEFAULT_DOWNLOAD_SETTINGS = DownloadSettings()
