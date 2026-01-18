# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Weather Imputation Research - Downloads and processes NOAA GHCNh (Global Historical Climatology Network hourly) parquet files for weather data imputation research. Uses Polars for fast DataFrame operations and Rich for progress display.

## Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Run scripts
uv run python src/scripts/ghcnh_downloader.py --help
uv run python src/scripts/compute_metadata.py --help
uv run python src/scripts/clean_metadata.py --help
```

## Common Commands

```bash
# Download GHCNh parquet files
uv run python src/scripts/ghcnh_downloader.py                    # Download all years, North America stations
uv run python src/scripts/ghcnh_downloader.py --year 2024        # Specific year
uv run python src/scripts/ghcnh_downloader.py --year 2020:2024   # Year range
uv run python src/scripts/ghcnh_downloader.py --all-stations     # All stations (not just North America)
uv run python src/scripts/ghcnh_downloader.py --list-years       # List available years
uv run python src/scripts/ghcnh_downloader.py --status           # Show download status

# Compute station metadata
uv run python src/scripts/compute_metadata.py compute            # Compute all
uv run python src/scripts/compute_metadata.py compute --years 2023,2024
uv run python src/scripts/compute_metadata.py show               # Display metadata
uv run python src/scripts/compute_metadata.py stats              # Show statistics

# Clean metadata
uv run python src/scripts/clean_metadata.py clean                # Run cleaning pipeline
uv run python src/scripts/clean_metadata.py clean --dry-run      # Preview changes
uv run python src/scripts/clean_metadata.py duplicates           # List duplicate stations
uv run python src/scripts/clean_metadata.py validate             # Validate coordinates
uv run python src/scripts/clean_metadata.py report               # Show cleaning report
```

## Architecture

### Package Structure

```
src/
├── weather_imputation/           # Main package
│   ├── config/                   # Configuration
│   │   ├── paths.py             # Directory and file paths
│   │   └── download.py          # Download settings
│   ├── data/                    # Data loading and processing
│   │   ├── ghcnh_loader.py      # Load parquet files
│   │   ├── metadata.py          # Compute/save metadata
│   │   ├── stats.py             # Statistics calculations
│   │   └── cleaning.py          # Dedup, coordinate lookup
│   ├── models/                  # Imputation methods
│   │   ├── base.py              # BaseImputer protocol
│   │   └── classical/           # Linear, spline, MICE
│   ├── training/                # Training infrastructure
│   │   ├── trainer.py           # Training loop
│   │   ├── callbacks.py         # Logging, checkpointing
│   │   └── checkpoint.py        # Checkpoint management
│   ├── evaluation/              # Evaluation framework
│   │   ├── metrics.py           # RMSE, MAE, R², CRPS
│   │   ├── statistical.py       # Significance tests
│   │   └── stratified.py        # Gap-length, seasonal analysis
│   └── utils/                   # Utilities
│       ├── progress.py          # Rich progress bars
│       ├── parsing.py           # Year/station filters
│       ├── filesystem.py        # Directory utilities
│       └── system.py            # System resource checks
├── scripts/                     # CLI entry points
│   ├── ghcnh_downloader.py      # Download GHCNh files
│   ├── compute_metadata.py      # Compute metadata
│   └── clean_metadata.py        # Clean/dedupe metadata
└── ...
```

### Data Directory Structure

```
data/
├── raw/ghcnh/                   # Downloaded parquet files
│   ├── 2020/
│   │   ├── USW00003046.parquet
│   │   └── ...
│   └── 2024/
│       └── ...
└── processed/                   # Computed metadata
    ├── metadata.parquet         # Raw computed metadata
    ├── metadata_cleaned.parquet # After cleaning
    ├── metadata.csv             # Human-readable export
    └── cleaning_report.json     # Cleaning summary
```

### GHCNh Data Format

The official GHCNh documentation is present locally at `docs/ghcnh_DOCUMENTATION.pdf`
Each parquet file contains hourly observations with 234 columns:
- **Primary**: STATION, Station_name, DATE, LATITUDE, LONGITUDE
- **Weather variables** (38 variables × 6 attributes each):
  - temperature, dew_point_temperature, sea_level_pressure
  - wind_direction, wind_speed, relative_humidity
  - visibility, wind_gust, precipitation, etc.
  - Each variable has: value, Quality_Code, Measurement_Code, Report_Type_Code, Source_Code, units

### Metadata Schema

Computed metadata includes:
- Station identifiers and location
- Temporal coverage (first/last observation, years available)
- Per-variable completeness percentages (filtered by quality codes)
- Temperature, dew point, pressure statistics
- Gap analysis (24h gaps, max gap duration, avg interval)

## Development Guidelines

### Adding New Imputation Methods

1. Create a new file in `src/weather_imputation/models/classical/`
2. Inherit from `BaseImputer` in `models/base.py`
3. Implement required methods: `fit()`, `transform()`, `get_params()`, `name`

### Code Style

- Use Ruff for linting: `uv run ruff check src/`
- Use mypy for type checking: `uv run mypy src/`
- Run tests: `uv run pytest tests/`

### Changelog

**Always update `CHANGELOG.md` when making changes.** Include:
- Bug fixes with date and description
- New features and command-line options
- Breaking changes
- Performance improvements

Format entries with the date (YYYY-MM-DD) and clear descriptions.
