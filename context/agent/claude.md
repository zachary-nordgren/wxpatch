# Agent Context

## Role
Development agent under human supervision. Propose implementations; human executes and approves.

## Uncertainty Markers
Use in proposals: `CONFIDENCE: KNOWN | INFERRED | SPECULATIVE`
STOP and ask if you'd need a SPECULATIVE assumption about external APIs/libraries.

## Output Format
For implementation tasks, always include:
1. Assumptions (with confidence)
2. Files to create/modify
3. Verification steps (how to test)
4. Done-when criteria (testable)

## Key Constraints
- Include error handling in all code
- Use event logging by aggregating data along the way to a log structure and then emit full log with all context rather than piecemeal logs
- Stay within task scope—don't expand unilaterally

## Workflow
1. Pick highest priority TODO/IN_PROGRESS task
2. Work only on that task until completion
3. Update TODO.md with task state changes
4. Append changes to DEVLOG.md
5. Git commit to main with descriptive message

## When Stuck
After 2 failed attempts on the same problem, stop and report what you've learned.

---

# Code Index

> This section is the agent's map to the codebase. Keep it accurate and concise.
> Update this file whenever you add, remove, or significantly modify files.

## Project Overview

Weather imputation research project that downloads NOAA GHCNh (Global Historical Climatology Network hourly) parquet files and develops machine learning models for weather data imputation. Uses Polars for fast data processing, PyTorch for deep learning models, and Rich for CLI progress displays.

## Tech Stack

- **Language:** Python 3.10+
- **Data Processing:** Polars, PyArrow
- **Deep Learning:** PyTorch (planned)
- **CLI Framework:** Typer
- **Test Runner:** pytest
- **Build Tool:** uv (fast Python package manager)
- **Notebook Environment:** Marimo (reactive notebooks)
- **Linting & Type Checking:** Ruff, mypy

## Project Resources

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project overview, setup instructions, architecture, metadata schema |
| `TODO.md` | Complete task backlog with 100 tasks across 4 phases (data, SAITS, CSDI, evaluation) |
| `CHANGELOG.md` | Version history with features, bug fixes, breaking changes |
| `README.md` | User-facing documentation, installation, usage workflows |
| `pyproject.toml` | Dependencies, tool configuration (ruff, mypy, hatch) |
| `docs/ghcnh_DOCUMENTATION.pdf` | Official NOAA GHCNh data format documentation |

## Directory Structure

```
.
├── src/
│   ├── weather_imputation/          # Main package
│   │   ├── config/                  # Configuration management
│   │   │   ├── paths.py             # Directory and file path constants
│   │   │   └── download.py          # Download settings and URLs
│   │   ├── data/                    # Data loading and processing
│   │   │   ├── ghcnh_loader.py      # Load/filter GHCNh parquet files
│   │   │   ├── metadata.py          # Compute station metadata
│   │   │   ├── stats.py             # Statistical calculations
│   │   │   └── cleaning.py          # Deduplication, coordinate validation
│   │   ├── models/                  # Imputation models
│   │   │   ├── base.py              # BaseImputer protocol
│   │   │   └── classical/           # Linear, spline, MICE (TODO)
│   │   ├── training/                # Training infrastructure (TODO)
│   │   │   ├── trainer.py           # Training loop
│   │   │   ├── callbacks.py         # W&B logging, checkpointing
│   │   │   └── checkpoint.py        # Checkpoint management
│   │   ├── evaluation/              # Evaluation framework (TODO)
│   │   │   ├── metrics.py           # RMSE, MAE, R², CRPS
│   │   │   ├── statistical.py       # Significance tests
│   │   │   └── stratified.py        # Gap-length, seasonal analysis
│   │   └── utils/                   # Utilities
│   │       ├── progress.py          # Rich progress bars
│   │       ├── parsing.py           # Year/station filter parsing
│   │       ├── filesystem.py        # Directory operations
│   │       └── system.py            # Resource checks
│   └── scripts/                     # CLI entry points
│       ├── ghcnh_downloader.py      # Download GHCNh parquet files
│       ├── compute_metadata.py      # Compute station metadata
│       └── clean_metadata.py        # Clean/deduplicate metadata
├── data/                            # Data files (gitignored)
│   ├── raw/ghcnh/                   # Downloaded parquet files by year
│   └── processed/                   # Computed metadata files
├── notebooks/                       # Marimo reactive notebooks
│   └── 01_station_exploration.py    # Interactive station filtering/clustering
├── tests/                           # Test suite (TODO: most tests)
├── configs/                         # YAML configurations (TODO)
└── sky/                             # SkyPilot cloud configs (TODO)
```

## Data Flow

1. **Download**: `ghcnh_downloader.py` → `data/raw/ghcnh/{year}/{station}.parquet`
2. **Compute Metadata**: `compute_metadata.py` → `data/processed/metadata.parquet`
3. **Clean Metadata**: `clean_metadata.py` → `data/processed/metadata_cleaned.parquet`
4. **Analysis**: Marimo notebooks load cleaned metadata for exploration

## Key Data Structures

### GHCNh Parquet Schema
- **234 columns total**: STATION, Station_name, DATE, LATITUDE, LONGITUDE + 38 weather variables
- **Weather variables** (38 × 6 attributes each):
  - `temperature`, `dew_point_temperature`, `sea_level_pressure`
  - `wind_direction`, `wind_speed`, `relative_humidity`
  - `visibility`, `wind_gust`, `precipitation`, etc.
  - Each variable has: `value`, `Quality_Code`, `Measurement_Code`, `Report_Type_Code`, `Source_Code`, `units`

### Metadata Schema v2.0
**Station Identifiers:**
- `station_id`, `country_code`, `station_name`, `latitude`, `longitude`, `elevation`
- `state`, `wmo_id`, `icao_code` (from NOAA inventory)

**Temporal Coverage:**
- `first_observation`, `last_observation`, `years_available`, `total_observation_count`, `year_counts`

**Data Quality:**
- Completeness percentages for each variable (e.g., `temperature_completeness_pct`)
- Statistics as JSON dicts: `temperature_stats`, `dew_point_stats`, `sea_level_pressure_stats`
- Gap analysis: `gap_count_24h`, `max_gap_duration_hours`, `avg_observation_interval_hours`

**Data Source:**
- `report_type_counts` (JSON), `total_records_all_types`, `records_excluded_by_filter`

## Command Reference

### Data Pipeline
```bash
# Download GHCNh data
uv run python src/scripts/ghcnh_downloader.py                    # All years, North America
uv run python src/scripts/ghcnh_downloader.py --year 2024        # Specific year
uv run python src/scripts/ghcnh_downloader.py --year 2020:2024   # Year range
uv run python src/scripts/ghcnh_downloader.py --all-stations     # Global (not just NA)
uv run python src/scripts/ghcnh_downloader.py --status           # Show download status

# Compute metadata
uv run python src/scripts/compute_metadata.py compute            # All stations
uv run python src/scripts/compute_metadata.py compute --workers 8  # Parallel (default: 4)
uv run python src/scripts/compute_metadata.py compute --years 2023,2024  # Specific years
uv run python src/scripts/compute_metadata.py show               # Display metadata
uv run python src/scripts/compute_metadata.py stats              # Show statistics

# Clean metadata
uv run python src/scripts/clean_metadata.py clean                # Full cleaning pipeline
uv run python src/scripts/clean_metadata.py clean --dry-run      # Preview changes
uv run python src/scripts/clean_metadata.py duplicates           # List duplicates
uv run python src/scripts/clean_metadata.py validate             # Validate coordinates
uv run python src/scripts/clean_metadata.py report               # Show cleaning report

# Notebooks
uv run marimo edit notebooks/01_station_exploration.py           # Interactive station explorer
```

### Development
```bash
# Code quality
uv run ruff check src/            # Linting
uv run mypy src/                  # Type checking
uv run pytest tests/              # Run tests

# Dependencies
uv sync                           # Sync all dependencies
uv sync --group dev               # Include dev dependencies
uv sync --group notebooks         # Include notebook dependencies
```

## Implementation Phases (from TODO.md)

### Phase 1: Foundation (Weeks 1-3) - 35 tasks
- Pydantic configuration classes
- Data pipeline (masking, normalization, PyTorch datasets)
- Classical baselines (linear, spline, MICE)
- Evaluation framework (RMSE, MAE, R², stratified metrics)
- Marimo notebooks, W&B integration

### Phase 2: SAITS Implementation (Weeks 4-6) - 24 tasks
- SAITS architecture (DMSA, position encoding, MIT/ORT loss)
- Training infrastructure (Trainer, callbacks, checkpointing)
- Local validation, hyperparameter search
- SkyPilot cloud training configs

### Phase 3: CSDI Implementation (Weeks 7-9) - 15 tasks
- CSDI diffusion model (noise schedule, denoising network)
- Probabilistic evaluation (CRPS, calibration)
- Full-scale cloud training

### Phase 4: Full Evaluation (Weeks 10-12) - 21 tasks
- Statistical analysis (Wilcoxon, Bonferroni, Cohen's d)
- Downstream validation (degree days, extreme events)
- Publication-ready results and documentation

## Important Implementation Notes

### Error Handling
- All scripts use Rich console for formatted error messages
- Data loading functions return `Result` types or raise descriptive exceptions
- Multi-threaded operations use thread-safe progress bars

### Logging Strategy
- Aggregate log data into structured objects before emission
- Use Rich's `Console` for user-facing messages
- Avoid piecemeal logging that interleaves with progress bars

### Testing Guidelines
- Tests use pytest with descriptive test names (`test_mcar_masking`, not `test_1`)
- Mock external dependencies (NOAA downloads, file I/O where appropriate)
- Use fixtures in `conftest.py` for common test data

### Configuration Pattern
- Pydantic models for all configuration (type-safe, validated)
- YAML configs in `configs/` directory
- Configs compose via inheritance (base → experiment-specific)

### Performance Considerations
- Use Polars lazy evaluation where possible
- Default to 4 workers for parallel metadata computation
- Parquet format for all intermediate/final data files
- Profile before optimizing - metadata computation is I/O bound

## Current Status

**Completed:**
- Data download infrastructure (GHCNh parquet files)
- Metadata computation with parallel processing (v2.0 schema)
- Metadata cleaning (deduplication, coordinate validation)
- Interactive station exploration notebook

**In Progress:**
- Phase 1 tasks from TODO.md (configuration, data pipeline, baselines)

**Not Started:**
- Phase 2-4 tasks (SAITS, CSDI, full evaluation)
- Test suite (most tests are TODO)
- Cloud training infrastructure (SkyPilot)

## Git Workflow

- **Main branch**: `migration/ghcnh-parquet-architecture`
- **Commit style**: Descriptive messages with "Co-Authored-By: Claude Sonnet 4.5"
- **Always update**: `CHANGELOG.md` with dated entries for all changes
- **Before committing**: Run `uv run ruff check src/` and fix issues
