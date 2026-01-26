# Weather Data Imputation Research Framework

A Python-based research framework for multi-variable weather imputation using NOAA GHCNh (Global Historical Climatology Network hourly) data. Implements classical baselines (linear, spline, MICE) and modern deep learning methods (SAITS attention-based, CSDI diffusion-based) with rigorous evaluation.

## Quick Start

```bash
# Install dependencies
uv sync

# Download GHCNh data
uv run python src/scripts/ghcnh_downloader.py

# Compute station metadata
uv run python src/scripts/compute_metadata.py compute

# Clean metadata
uv run python src/scripts/clean_metadata.py clean

# Explore stations interactively
uv run marimo edit notebooks/01_station_exploration.py
```

## Project Overview

This framework addresses how researchers can systematically compare classical and deep learning imputation methods on real-world hourly weather data within a $100-300 compute budget. It processes 3,000-6,000 North American weather stations from GHCNh parquet files and evaluates 5 imputation methods with statistical significance testing and downstream validation.

See **[SPEC.md](SPEC.md)** for complete project specification, requirements, architecture, and technical decisions.

## Tech Stack

- **Language:** Python 3.10+ with type hints (mypy strict mode)
- **Data Processing:** Polars (5-10x faster than Pandas, lazy evaluation)
- **Deep Learning:** PyTorch 2.0+ (torch.compile, SDPA)
- **Notebooks:** Marimo (reactive, git-friendly pure Python files)
- **CLI:** Typer (type-safe command-line interfaces)
- **Configuration:** Pydantic v2 (type-validated YAML/JSON)
- **Testing:** pytest with descriptive test names
- **Build Tool:** uv (10-100x faster than pip)
- **Linting:** Ruff (Rust-based, auto-fix)
- **Cloud:** SkyPilot (multi-cloud spot instances with auto-recovery)
- **Tracking:** Weights & Biases (experiment logging, artifact storage)

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
│   │   │   ├── cleaning.py          # Deduplication, coordinate validation
│   │   │   ├── masking.py           # Synthetic gap generation (TODO)
│   │   │   ├── normalization.py     # Per-station z-score (TODO)
│   │   │   ├── dataset.py           # PyTorch datasets (TODO)
│   │   │   └── splits.py            # Train/val/test splitting (TODO)
│   │   ├── models/                  # Imputation models
│   │   │   ├── base.py              # BaseImputer protocol
│   │   │   ├── classical/           # Linear, spline, MICE (TODO)
│   │   │   ├── attention/           # SAITS (TODO)
│   │   │   └── diffusion/           # CSDI (TODO)
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
│       ├── clean_metadata.py        # Clean/deduplicate metadata
│       ├── preprocess.py            # Data preprocessing (TODO)
│       ├── train.py                 # Model training (TODO)
│       └── evaluate.py              # Model evaluation (TODO)
├── data/                            # Data files (gitignored)
│   ├── raw/ghcnh/                   # Downloaded parquet files by year
│   └── processed/                   # Computed metadata files
├── notebooks/                       # Marimo reactive notebooks
│   ├── 01_station_exploration.py    # Interactive station filtering
│   ├── 02_data_quality.py           # Quality analysis (TODO)
│   ├── 03_model_comparison.py       # Model training notebook (TODO)
│   └── 04_results_analysis.py       # Publication figures (TODO)
├── tests/                           # Test suite (TODO: most tests)
├── configs/                         # YAML configurations (TODO)
├── checkpoints/                     # Trained model checkpoints (TODO)
├── results/                         # Evaluation outputs (TODO)
└── sky/                             # SkyPilot cloud configs (TODO)
```

## Data Flow

### 1. Setup Phase
```
ghcnh_downloader.py → data/raw/ghcnh/{year}/{station}.parquet
compute_metadata.py → data/processed/metadata.parquet
clean_metadata.py   → data/processed/metadata_cleaned.parquet
```

### 2. Exploration Phase
```
metadata_cleaned.parquet → 01_station_exploration.py (Marimo)
User adjusts thresholds → selected_stations.json
```

### 3. Preprocessing Phase (TODO)
```
preprocess.py:
  - Load selected_stations.json
  - For each station: load multi-year data → filter → normalize
  - Apply train/val/test splits (Strategy D: simulated masks)
  - Save: train.parquet, val.parquet, test.parquet
```

### 4. Training Phase (TODO)
```
train.py:
  - Load config (model + training hyperparameters)
  - Initialize Dataset, DataLoader, Model (SAITS/CSDI)
  - Trainer.train() → checkpoints/exp_{id}/checkpoint_epoch_N.pt
  - Log to W&B: loss curves, val metrics
```

### 5. Evaluation Phase (TODO)
```
evaluate.py:
  - Load checkpoint → model.load()
  - Compute predictions on test set
  - Compute metrics (point + probabilistic + stratified)
  - Statistical tests → results/comparison_table.csv
```

### 6. Publication Phase (TODO)
```
04_results_analysis.py (Marimo):
  - Generate publication-quality figures
  - Export notebook → notebooks/exports/results_YYYYMMDD.html
```

## Command Reference

### Data Pipeline

```bash
# Download GHCNh parquet files
uv run python src/scripts/ghcnh_downloader.py                    # All years, North America
uv run python src/scripts/ghcnh_downloader.py --year 2024        # Specific year
uv run python src/scripts/ghcnh_downloader.py --year 2020:2024   # Year range
uv run python src/scripts/ghcnh_downloader.py --all-stations     # Global (not just NA)
uv run python src/scripts/ghcnh_downloader.py --status           # Show download status

# Compute station metadata
uv run python src/scripts/compute_metadata.py compute            # All stations
uv run python src/scripts/compute_metadata.py compute --workers 8  # Parallel (default: 4)
uv run python src/scripts/compute_metadata.py compute --years 2023,2024  # Specific years
uv run python src/scripts/compute_metadata.py show               # Display metadata
uv run python src/scripts/compute_metadata.py stats              # Show statistics

# Clean metadata (deduplication, coordinate validation)
uv run python src/scripts/clean_metadata.py clean                # Full cleaning pipeline
uv run python src/scripts/clean_metadata.py clean --dry-run      # Preview changes
uv run python src/scripts/clean_metadata.py duplicates           # List duplicates
uv run python src/scripts/clean_metadata.py validate             # Validate coordinates
uv run python src/scripts/clean_metadata.py report               # Show cleaning report
```

### Notebooks

```bash
# Interactive station exploration (filtering, clustering)
uv run marimo edit notebooks/01_station_exploration.py

# Run notebook in read-only mode
uv run marimo run notebooks/01_station_exploration.py
```

### Development

```bash
# Code quality
uv run ruff check src/            # Linting
uv run ruff check src/ --fix      # Auto-fix issues
uv run mypy src/                  # Type checking
uv run pytest tests/              # Run tests

# Dependencies
uv sync                           # Sync all dependencies
uv sync --group dev               # Include dev dependencies
uv sync --group notebooks         # Include notebook dependencies
uv add <package>                  # Add new dependency
```

## Data Schemas

### GHCNh Parquet Schema

Each station parquet file contains:
- **234 columns total**: STATION, Station_name, DATE, LATITUDE, LONGITUDE + 38 weather variables
- **Weather variables** (38 × 6 attributes each):
  - Core variables: `temperature`, `dew_point_temperature`, `sea_level_pressure`, `wind_direction`, `wind_speed`, `relative_humidity`
  - Extended variables: `visibility`, `wind_gust`, `precipitation`, `snow_depth`, etc.
  - Each variable has: `value`, `Quality_Code`, `Measurement_Code`, `Report_Type_Code`, `Source_Code`, `units`

### Metadata Schema v2.0

**Station Identifiers:**
- `station_id`, `country_code`, `station_name`, `latitude`, `longitude`, `elevation`
- `state`, `wmo_id`, `icao_code` (from NOAA inventory)

**Temporal Coverage:**
- `first_observation`, `last_observation`, `years_available`
- `total_observation_count`, `year_counts` (JSON dict)

**Data Quality:**
- Completeness percentages: `temperature_completeness_pct`, `dew_point_completeness_pct`, etc.
- Statistics as JSON dicts: `temperature_stats`, `dew_point_stats`, `sea_level_pressure_stats`
- Gap analysis: `gap_count_24h`, `max_gap_duration_hours`, `avg_observation_interval_hours`

**Data Source:**
- `report_type_counts` (JSON), `total_records_all_types`, `records_excluded_by_filter`

See [SPEC.md §3.4](SPEC.md#34-data-model) for complete schema definitions.

## Implementation Phases

The project is organized into 4 phases over 12 weeks (part-time). See **[TODO.md](TODO.md)** for complete task backlog with priorities.

### Phase 1: Foundation (Weeks 1-3) - 35 tasks
- ✅ Data download infrastructure (GHCNh parquet files)
- ✅ Metadata computation with parallel processing
- ✅ Metadata cleaning (deduplication, coordinate validation)
- ✅ Interactive station exploration notebook
- 🚧 Pydantic configuration classes
- 🚧 Data pipeline (masking, normalization, PyTorch datasets)
- 🚧 Classical baselines (linear, spline, MICE)
- 🚧 Evaluation framework (RMSE, MAE, R², stratified metrics)

### Phase 2: SAITS Implementation (Weeks 4-6) - 24 tasks
- SAITS architecture (DMSA, position encoding, MIT/ORT loss)
- Training infrastructure (Trainer, callbacks, checkpointing)
- Weather-specific adaptations (circular wind encoding, metadata conditioning)
- Local validation, hyperparameter search

### Phase 3: CSDI Implementation (Weeks 7-9) - 15 tasks
- CSDI diffusion model (noise schedule, denoising network)
- Probabilistic evaluation (CRPS, calibration)
- Full-scale cloud training with SkyPilot

### Phase 4: Full Evaluation (Weeks 10-12) - 21 tasks
- Statistical analysis (Wilcoxon, Bonferroni, Cohen's d)
- Downstream validation (degree days, extreme events)
- Publication-ready results and documentation

## Implementation Guidelines

### Error Handling
- All scripts use Rich console for formatted error messages
- Data loading functions raise descriptive exceptions with context
- Multi-threaded operations use thread-safe progress bars
- Aggregate log data into structured objects before emission (see [SPEC.md §5.2](SPEC.md#52-logging-strategy))

### Testing
- Tests use pytest with descriptive test names (`test_mcar_masking_preserves_marginal_distribution`)
- Mock external dependencies (NOAA downloads, W&B API, file I/O where appropriate)
- Use fixtures in `conftest.py` for reusable test data
- Tests mirror source structure: `tests/test_data/test_masking.py`

### Configuration Pattern
- Pydantic models for all configuration (type-safe, validated)
- YAML configs in `configs/` directory
- Configs compose via inheritance (base → experiment-specific)

### Performance
- Use Polars lazy evaluation where possible
- Default to 4 workers for parallel metadata computation (configurable)
- Parquet format for all intermediate/final data files
- Profile before optimizing - metadata computation is I/O bound

## Project Status

**Completed:**
- ✅ Data download infrastructure
- ✅ Metadata computation (v2.0 schema)
- ✅ Metadata cleaning
- ✅ Station exploration notebook

**In Progress:**
- 🚧 Phase 1 foundation tasks (configuration, data pipeline, baselines)

**Not Started:**
- ⬜ Phase 2-4 tasks (SAITS, CSDI, full evaluation)
- ⬜ Test suite (most tests are TODO)
- ⬜ Cloud training infrastructure

## Resources

### Documentation
- **[SPEC.md](SPEC.md)** - Complete project specification (requirements, architecture, API contracts)
- **[TODO.md](TODO.md)** - Task backlog with 100 tasks across 4 phases
- **[DEVLOG.md](DEVLOG.md)** - Development log with dated entries
- **[CLAUDE.md](CLAUDE.md)** - Agent context and workflow
- **[docs/ghcnh_DOCUMENTATION.pdf](docs/ghcnh_DOCUMENTATION.pdf)** - Official NOAA data format

### Papers
- **SAITS:** Du et al. (2023) "SAITS: Self-Attention-based Imputation for Time Series" ([arXiv](https://arxiv.org/abs/2202.08516))
- **CSDI:** Tashiro et al. (2021) "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation" ([NeurIPS](https://arxiv.org/abs/2107.03502))
- **MICE:** van Buuren & Groothuis-Oudshoorn (2011) "mice: Multivariate Imputation by Chained Equations in R" ([JSS](https://www.jstatsoft.org/article/view/v045i03))

### External Links
- **NOAA GHCNh Parquet Files:** [S3 Bucket](https://noaa-ghcn-pds.s3.amazonaws.com/index.html)
- **Marimo:** [marimo.io](https://marimo.io)
- **Pydantic:** [docs.pydantic.dev](https://docs.pydantic.dev/latest/)
- **SkyPilot:** [skypilot.readthedocs.io](https://skypilot.readthedocs.io)
- **Weights & Biases:** [docs.wandb.ai](https://docs.wandb.ai)

## License

[Specify license here]
