# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NOAA Weather Data Processor - Downloads and processes NOAA Global Hourly weather data to create a complete, accurate historical weather dataset for the continental USA. Uses Polars for fast CSV/Parquet operations with multi-threaded processing.

## Common Commands

```bash
# Install dependencies
pip install requests tqdm polars psutil

# Full download and process (from src/ directory)
python main.py

# Download only (no processing)
python main.py --download-only

# Process already-downloaded archives
python main.py --process-only

# Process specific years
python main.py --year-filter 2020,2021,2022
python main.py --year-filter 2010:2015,2020  # Range plus single year

# Process newest years first
python main.py --newest-first

# Update dataset from last observation to present
python update_dataset.py
python update_dataset.py --no-stats  # Skip statistics (faster)

# Combine two merged datasets
python combine_datasets.py
python combine_datasets.py --parquet  # Output as Parquet

# Sort all station files chronologically
python main.py --sort-chronologically

# Control parallelism (default: 14 workers)
python main.py --max-workers 20

# Clean up (preserves raw downloads)
python main.py --clean

# Recover corrupted metadata
python main.py --recover-metadata
```

## Architecture

### Entry Points
- `main.py` - Main orchestrator for downloading and processing
- `update_dataset.py` - Incremental updates from last observation date to present
- `combine_datasets.py` - Merge two dataset directories into one

### Core Processing Pipeline
1. **downloader.py** - Fetches tar.gz archives from NOAA servers
2. **processor_core.py** - Producer/consumer pattern for archive processing
   - `StationProcessor` class handles extraction and processing threads
   - Extraction workers read from tar archives into queue
   - Processing workers consume queue and merge station data
3. **csv_merger.py** - Filters CSV data and merges with Polars
4. **file_io.py** - Directory setup and file modification tracking

### Metadata System
- `metadata_manager.py` - SQLite-based metadata storage with thread-safe connections
- Database at `data/merged/wx_metadata.db` with stations and year_counts tables
- Exports to `wx_info.csv` for compatibility
- Statistics include: completeness percentages, temperature/pressure stats, gap analysis

### Configuration
- `config.py` - All paths, URLs, and tunable parameters
- Data directories relative to `src/`: `../data/raw/`, `../data/merged/`, `../data/temp/`

### Key Data Flow
```
NOAA Archives (.tar.gz) → Extract to temp → Filter CSV → Merge by station → Compress (csv.gz/parquet)
                                                      → Update SQLite metadata
```

### Parallelism Model
- Archives processed sequentially
- Files within each archive processed concurrently via producer/consumer queues
- Fixed extraction workers (1) + configurable processing workers (default: 13)
- Thread-local SQLite connections with semaphore limiting (max 3 concurrent)

## Output Format

Station data files: `data/merged/{station_id}.csv.gz` or `.parquet`
- DATE column in ISO format: `YYYY-MM-DDTHH:MM:SS`
- Weather fields contain value,quality,source format (e.g., TMP: `215,1,1`)

## Logging

Log file: `src/weather_data_processing.log`
Use `--verbose` for debug output, `--quiet` for warnings only.

## Development Guidelines

### Changelog
**Always update `CHANGELOG.md` when making changes to the codebase.** Include:
- Bug fixes with date and description
- New features and command-line options
- Breaking changes
- Performance improvements

Format entries with the date (YYYY-MM-DD) and clear descriptions of what changed and why.
