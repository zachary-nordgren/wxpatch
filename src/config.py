#!/usr/bin/env python3
"""
Configuration settings for the NOAA weather data processor.
"""
from pathlib import Path

# URLs and remote data sources
BASE_URL = "https://www.ncei.noaa.gov/data/global-hourly/archive/csv/"

# Directory structure
DATA_DIR = Path("../data")
RAW_DIR = DATA_DIR / "raw"
MERGED_DIR = DATA_DIR / "merged"
TEMP_DIR = DATA_DIR / "temp"

# File I/O settings
IO_BUFFER_SIZE = 16 * 1024 * 1024  # MB

# Fields for station metadata (used in wx_info.csv)
METADATA_FIELDS = ["STATION", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME", "CALL_SIGN"]

# Weather observation fields to exclude from station data
# These are stored once in the metadata file
METADATA_ONLY_FIELDS = ["LATITUDE", "LONGITUDE", "ELEVATION", "NAME"]

# Network settings
DOWNLOAD_TIMEOUT = 300  # seconds
DOWNLOAD_CHUNK_SIZE = 8192  # bytes
MAX_DOWNLOAD_RETRIES = 3  # Maximum number of download retry attempts
MAX_DOWNLOAD_THREADS = 4  # Maximum number of concurrent downloads

# Default processing settings
DEFAULT_MAX_WORKERS = 14  # Default number of parallel processing workers
METADATA_BATCH_SIZE = 1024  # Number of stations to process before writing metadata

# Producer/Consumer pattern settings
QUEUE_SIZE = 200  # Maximum number of files in the queue
EXTRACTION_WORKER_NUMBER = 1  # Fixed number of workers to use for extraction
PROGRESS_LOG_INTERVAL = 100  # Log progress after this many files
MIN_VALID_LINE_LENGTH = 10  # Minimum length for a valid metadata line

# Logging settings
LOG_FILE = "weather_data_processing.log"

# Command line options
DOWNLOAD_ONLY_OPTION = "--download-only"
PROCESS_ONLY_OPTION = "--process-only"
MAX_WORKERS_OPTION = "--max-workers"
YEAR_FILTER_OPTION = "--year-filter"
UPDATE_COUNTS_OPTION = "--update-counts"
RECOVER_METADATA_OPTION = "--recover-metadata"
VERBOSE_OPTION = "--verbose"
QUIET_OPTION = "--quiet"
NEWEST_FIRST_OPTION = "--newest-first"
SORT_CHRONOLOGICALLY_OPTION = "--sort-chronologically"
