# Weather Data Imputation: Software Development Plan (v2)

## Overview

This document outlines a development plan for implementing a weather data imputation research framework using the GHCNh (Global Historical Climatology Network hourly) dataset. The project employs a modern Python stack consisting of Marimo for interactive exploration, Pydantic for configuration management, PyTorch for model implementation, and SkyPilot for cost-efficient cloud training.

**Data Source**: GHCNh parquet files from NOAA (~8,787 North American stations)  
**Budget**: $100-300 for cloud compute  
**Timeline**: 2-3 months, part-time effort  
**Scope**: Multi-variable imputation with classical baselines (linear, spline, MICE) and modern deep learning methods (SAITS, CSDI)

---

## 1. Technology Stack Rationale

### 1.1 Marimo (Notebooks)

Marimo replaces Jupyter for all exploratory and documentation work. Key advantages for this project include reactive execution ensuring cell outputs always reflect current code, pure Python file storage enabling proper version control, built-in UI components (sliders, dropdowns) for interactive threshold exploration, and straightforward export to HTML for archiving experiment results. All exploration notebooks will be stored in the repository and can serve as supplementary material for any future publication.

### 1.2 Pydantic (Configuration)

Pydantic handles all configuration through validated dataclasses. This provides type safety preventing silent configuration errors, JSON schema generation for documentation, seamless serialization for experiment tracking, and natural integration with SkyPilot's YAML configurations. The configuration hierarchy will mirror the project structure: data configs, model configs, training configs, and evaluation configs.

### 1.3 PyTorch (Models)

PyTorch remains the standard for research implementations. The codebase will target PyTorch 2.0+ to leverage torch.compile for performance improvements and the improved memory efficiency of the scaled_dot_product_attention implementation.

### 1.4 SkyPilot (Cloud Orchestration)

SkyPilot provides cloud-agnostic job management with automatic spot instance handling. For this project's budget constraints, SkyPilot offers automatic spot instance provisioning across AWS, GCP, Azure, and Lambda Labs; built-in checkpointing with automatic job restart on preemption; cost optimization through spot instance bidding and provider selection; and simple YAML-based job definitions that integrate with the Pydantic configs. A typical training job can be launched with `sky launch train.yaml` and SkyPilot handles the rest.

---

## 2. Revised Variable Selection

Based on the GHCNh documentation, the following variables are recommended for imputation modeling. These are organized into tiers based on data availability, physical interpretability, and imputation difficulty.

### 2.1 Tier 1: Core Continuous Variables (Primary Focus)

| Variable | Units | Notes |
|----------|-------|-------|
| temperature | °C (tenths) | Most complete, strong diurnal/seasonal patterns |
| dew_point_temperature | °C (tenths) | Physically coupled to temperature |
| sea_level_pressure | hPa | Standardized across elevations, smoother spatial gradients |
| wind_speed | m/s | Higher variability, shorter correlation lengths |
| wind_direction | degrees | Circular variable requiring special handling |
| relative_humidity | % | Often derived from temp/dewpoint, but directly measured at some stations |

### 2.2 Tier 2: Secondary Continuous Variables

| Variable | Units | Notes |
|----------|-------|-------|
| visibility | km | Important for aviation, affected by many phenomena |
| wind_gust | m/s | Sparse, intermittent by nature |
| wet_bulb_temperature | °C (tenths) | Often derived, increasingly important for heat stress |
| altimeter | hPa | Alternative pressure measurement |
| station_level_pressure | hPa | Elevation-dependent, useful with metadata |

### 2.3 Tier 3: Categorical/Complex Variables (Future Work)

| Variable | Type | Notes |
|----------|------|-------|
| sky_cover_1/2/3 | Categorical (oktas) | Ordinal categories, cloud layer structure |
| sky_cover_baseht_1/2/3 | Continuous (meters) | Conditional on sky_cover presence |
| precipitation | mm | Accumulation semantics, zero-inflated distribution |
| snow_depth | mm | Seasonal, regional |
| pres_wx_* | Categorical codes | Present weather, complex encoding |

### 2.4 Recommended Initial Variable Set

For the initial implementation, focus on the six Tier 1 variables. This matches the original thesis scope while providing enough complexity to validate the framework. Wind direction requires circular statistics (e.g., encoding as sin/cos components or using von Mises distributions for probabilistic methods).

The configuration should allow easy addition of Tier 2 variables once the pipeline is validated.

---

## 3. Project Structure

```
weather-imputation/
├── src/
│   ├── weather_imputation/        # Main package
│   │   ├── __init__.py
│   │   ├── config/                # Pydantic configuration models
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Base config classes
│   │   │   ├── data.py            # DataConfig, StationFilterConfig
│   │   │   ├── model.py           # ModelConfig variants
│   │   │   ├── training.py        # TrainingConfig
│   │   │   └── evaluation.py      # EvaluationConfig
│   │   │
│   │   ├── data/                  # Data loading and preprocessing
│   │   │   ├── __init__.py
│   │   │   ├── ghcnh_loader.py    # GHCNh-specific loading utilities
│   │   │   ├── dataset.py         # PyTorch Dataset classes
│   │   │   ├── masking.py         # Gap generation strategies
│   │   │   ├── normalization.py   # Per-variable normalization
│   │   │   └── splits.py          # Train/val/test splitting
│   │   │
│   │   ├── models/                # Imputation methods
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # BaseImputer protocol
│   │   │   ├── classical/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── linear.py
│   │   │   │   ├── spline.py
│   │   │   │   └── mice.py
│   │   │   ├── attention/
│   │   │   │   ├── __init__.py
│   │   │   │   └── saits.py
│   │   │   └── diffusion/
│   │   │       ├── __init__.py
│   │   │       └── csdi.py
│   │   │
│   │   ├── training/              # Training infrastructure
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py         # Main training loop
│   │   │   ├── callbacks.py       # Checkpointing, logging
│   │   │   └── checkpoint.py      # SkyPilot-compatible checkpointing
│   │   │
│   │   ├── evaluation/            # Evaluation framework
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py         # RMSE, MAE, CRPS, etc.
│   │   │   ├── statistical.py     # Significance tests
│   │   │   └── stratified.py      # Gap-length, seasonal analysis
│   │   │
│   │   └── utils/                 # Utilities
│   │       ├── __init__.py
│   │       ├── logging.py         # W&B integration
│   │       ├── circular.py        # Wind direction utilities
│   │       └── reproducibility.py # Seed management
│   │
│   └── scripts/                   # Entry points
│       ├── train.py               # Training CLI
│       ├── evaluate.py            # Evaluation CLI
│       ├── ghcnh_downloader.py    # GHCNh parquet files downloading and updating
│       └── preprocess.py          # Data preprocessing
│
├── notebooks/                     # Marimo notebooks
│   ├── 01_station_exploration.py  # Station filtering analysis
│   ├── 02_gap_analysis.py         # Missing data patterns
│   ├── 03_model_development.py    # Interactive model testing
│   ├── 04_results_analysis.py     # Final results visualization
│   └── exports/                   # HTML exports for archiving
│
├── configs/                       # Default configurations (YAML/JSON)
│   ├── data/
│   │   ├── north_america.yaml
│   │   └── regional_test.yaml
│   ├── model/
│   │   ├── linear.yaml
│   │   ├── spline.yaml
│   │   ├── mice.yaml
│   │   ├── saits.yaml
│   │   └── csdi.yaml
│   └── experiment/
│       ├── baseline_comparison.yaml
│       └── ablation_study.yaml
│
├── sky/                           # SkyPilot configurations
│   ├── train.yaml                 # Training job template
│   ├── evaluate.yaml              # Evaluation job template
│   └── setup.sh                   # Environment setup script
│
├── tests/                         # Unit tests
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_metrics.py
│
├── data/                          # Data directory (gitignored)
│   ├── raw/                       # Downloaded GHCNh parquet files
│   ├── processed/                 # Preprocessed data
│   └── results/                   # Experiment outputs
│
├── pyproject.toml                 # Project dependencies
├── CLAUDE.md                      # Development guidance
├── CHANGELOG.md                   # Change log
└── README.md                      # Project documentation
```

---

## 4. Data Pipeline

### 4.1 GHCNh Data Integration

The GHCNh format provides individual parquet files per station per year, which simplifies the data loading compared to the original ISD tar archives. The data pipeline consists of three stages.

**Stage 1: Inventory and Downloading**. Download all north American stations to data/raw/<year> folders and then calculate all stats and merge with inventory data to create a station metadata store `wx_info` stored as a parquet file (with a human readable csv as a copy).

**Stage 2: Quality Analysis (Marimo Notebook)**. For each station, filter out unwanted report types, then compute completeness statistics per variable, gap length distributions, temporal coverage, and quality flag frequencies. This analysis informs the filtering criteria through interactive exploration rather than predetermined thresholds.

**Stage 3: Preprocessing**. For stations passing the quality filter, extract relevant variables from the year-by-year parquet files, apply quality flag filtering (exclude values with erroneous QC codes), compute per-station normalization statistics, and save processed data in a format optimized for training (consolidated parquet or memory-mapped arrays).

### 4.2 Proposed Filtering Criteria (To Be Refined)

The following criteria serve as a starting point for the interactive exploration in the Marimo notebook. The goal is to balance data quality against station count, targeting approximately 3,000-6,000 stations.

| Criterion | Initial Threshold | Adjustable Range |
|-----------|-------------------|------------------|
| Temperature completeness | ≥70% | 50-90% |
| Minimum time span | 5 years | 3-10 years |
| Minimum observations | 20,000 | 10,000-50,000 |
| Valid coordinates | lat/lon not 0,0 | Fixed |
| Continental US/Canada/Mexico | Country code filter | Adjustable |
| Report Type | AUTO, FM-15 | Likely more as well |

The notebook will visualize histograms of these metrics and allow interactive threshold adjustment to see the resulting station count and geographic coverage.

### 4.3 Train/Validation/Test Splits

Strategy A: Held-out stations (spatial cross-validation)


	All years, all stations:
	├── Train: 70% of stations (randomly selected)
	├── Val:   15% of stations
	└── Test:  15% of stations

- Train/val/test on same time period
- Tests generalization to new locations
- Stratify by region/climate to ensure coverage

Strategy B: Temporal block splitting prevents information leakage:

| Split | Years | Approximate % |
|-------|-------|---------------|
| Train | 1973-2015 | 80% |
| Validation | 2016-2018 | 7% |
| Test | 2019-2023 | 13% |

Stations are not split; all stations appear in all temporal splits. This tests generalization across time rather than across stations.

Strategy C: Hybrid temporal + spatial


	For each fold:
	├── Hold out 2 random years (e.g., 2005, 2013) across all stations
	└── Use remaining years for training

- Prevents temporal bias
- Tests on same station in different time period
- More realistic: "I have data for this station, but gaps in year X"

Strategy D: Simulate Real Missing

- Within each signal, mask random values/blocks of values according to MCAR/MAR/MNAR/realistic-gap generator.
- Make two (non overlapping) masks, one for val and one for test. The data from both of those masks is removed from the Train dataset to avoid bleed.
- Train on the rest of the data (including the rest of those years).


---

## 5. Model Implementations

### 5.1 Common Interface

All imputation methods implement a common protocol defined using Python's typing.Protocol:

```python
class Imputer(Protocol):
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None: ...
    def impute(self, observed: Tensor, mask: Tensor) -> Tensor: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

Classical methods implement `fit` as a no-op. Neural methods implement full training loops.

### 5.2 Classical Methods

**Linear Interpolation**: Per-variable linear interpolation using numpy. Handles edge cases (gaps at sequence boundaries) by forward/backward filling. Serves as the simplest baseline.

**Akima Spline**: Uses scipy.interpolate.Akima1DInterpolator for smoother interpolation that avoids overshoot. Falls back to linear when insufficient points exist.

**MICE**: Wraps sklearn.impute.IterativeImputer with BayesianRidge estimator. Configured for 10 imputation rounds with convergence monitoring. Can optionally return multiple imputations for uncertainty estimation.

### 5.3 SAITS (Self-Attention Imputation for Time Series)

The SAITS implementation follows the architecture from Du et al. (2023) with weather-specific adaptations.

**Architecture**: Two stacked DMSA (Diagonally-Masked Self-Attention) blocks with 4 attention heads and 128 hidden dimensions (configurable). Position encoding uses learnable embeddings plus sinusoidal time-of-day and day-of-year encodings.

**Training**: Joint optimization of MIT (Masked Imputation Task) and ORT (Observed Reconstruction Task) losses. Mixed precision training enabled by default.

**Weather Adaptations**: Wind direction encoded as (sin θ, cos θ) pair. Cross-variable attention captures physical relationships. Optional conditioning on station metadata (elevation, latitude).

### 5.4 CSDI (Conditional Score-based Diffusion)

The CSDI implementation follows Tashiro et al. (2021) for probabilistic imputation.

**Architecture**: Transformer-based denoising network with 4 layers and 128 hidden dimensions. Conditioning mechanism concatenates observed values and mask with noised missing values.

**Diffusion Process**: 50 steps with cosine noise schedule. Self-supervised training masks additional observed values as imputation targets.

**Inference**: Generates 50-100 samples (configurable based on compute budget). Returns mean, standard deviation, and optionally full sample set.

**Note**: CSDI is significantly more expensive than SAITS (approximately 5-10x training time, 50-100x inference time). Budget allocation should account for this.

---

## 6. Evaluation Framework

### 6.1 Metrics

**Point Metrics** (all methods):
- RMSE: Root mean squared error, emphasizes large errors
- MAE: Mean absolute error, robust to outliers
- Bias: Mean signed error, detects systematic over/under-estimation

**Probabilistic Metrics** (CSDI only):
- CRPS: Continuous Ranked Probability Score
- Calibration: Observed coverage of prediction intervals

**Stratified Analysis**:
- By gap length: 1-6h, 7-24h, 25-72h, >72h
- By season: DJF, MAM, JJA, SON
- By variable: Separate metrics per variable
- By extreme values: Performance on values beyond 5th/95th percentiles

### 6.2 Statistical Significance

Method comparisons include paired Wilcoxon signed-rank tests (non-parametric), Bonferroni correction for multiple comparisons, and effect size reporting (Cohen's d). Results tables will include 95% confidence intervals computed via bootstrap.

### 6.3 Logging and Tracking

Weights & Biases (W&B) provides experiment tracking with the free tier sufficient for this project's scope. All training runs log hyperparameters (from Pydantic configs), training/validation loss curves, evaluation metrics at checkpoints, and system metrics (GPU utilization, memory). W&B's comparison features enable systematic analysis across experiments.

---

## 7. Cloud Strategy with SkyPilot

### 7.1 Cost Estimates

Based on current spot instance pricing and estimated training times:

| Task | GPU | Hours | Spot $/hr | Est. Cost |
|------|-----|-------|-----------|-----------|
| SAITS (regional, 100 epochs) | T4 | 15-20 | $0.12 | $2-3 |
| SAITS (full, 100 epochs) | A10 | 40-60 | $0.50 | $20-30 |
| CSDI (regional, 100 epochs) | T4 | 50-70 | $0.12 | $6-9 |
| CSDI (full, 100 epochs) | A10 | 150-200 | $0.50 | $75-100 |
| Evaluation (all methods) | T4 | 10-15 | $0.12 | $1-2 |
| **Total (conservative)** | | | | **$100-150** |

These estimates assume efficient checkpointing and minimal wasted compute from preemptions. The budget of $100-300 provides comfortable margin for experimentation and reruns.

### 7.2 SkyPilot Configuration

Example training job configuration (`sky/train.yaml`):

```yaml
name: weather-imputation-train

resources:
  cloud: aws  # or gcp, azure, lambda
  accelerators: T4:1  # or A10:1 for larger runs
  use_spot: true
  spot_recovery: failover  # Automatically restart on preemption

file_mounts:
  /data:
    source: s3://weather-imputation-data/processed/
    mode: COPY

setup: |
  pip install -e .
  pip install wandb

run: |
  cd /weather-imputation
  python scripts/train.py \
    --config configs/experiment/saits_regional.yaml \
    --checkpoint-dir /data/checkpoints \
    --wandb-project weather-imputation
```

### 7.3 Data Storage Strategy

For the $100-300 budget, avoid storing the full dataset on cloud storage long-term. Instead, store processed/filtered data only (estimated 5-20GB after filtering), use SkyPilot's file_mounts to sync data at job start, and keep raw GHCNh files local (they can be re-downloaded if needed).

Cloud storage costs for 20GB:
- S3/GCS: ~$0.50/month
- Negligible compared to compute costs

### 7.4 Checkpointing Strategy

Checkpoints save every 30 minutes and at each epoch boundary. Each checkpoint includes model state dict, optimizer state dict, scheduler state, current epoch and step, RNG states for reproducibility, and best validation metric seen. Checkpoints are saved to cloud storage (S3/GCS) to survive instance preemption. SkyPilot's spot_recovery automatically restarts from the latest checkpoint.

---

## 8. Downstream Validation Options

Since the original natural gas demand data is unavailable, consider these alternatives for downstream validation.

### 8.1 Option A: Renewable Energy Data (Recommended)

The NSRDB (National Solar Radiation Database) and wind resource datasets are publicly available and directly relevant to weather. Task: Predict solar irradiance or wind power potential from imputed weather variables. Data source: NREL (National Renewable Energy Laboratory) open datasets. Relevance: Temperature, humidity, and pressure affect solar panel efficiency; wind speed obviously affects wind power.

### 8.2 Option B: Agricultural Degree Days

Growing degree days and heating/cooling degree days are standard agricultural and energy metrics derived from temperature. Task: Compute degree day accumulations from imputed vs. original temperature series. Data source: Computed directly from the weather data itself. Relevance: Tests whether imputation preserves the statistical properties needed for agricultural planning.

### 8.3 Option C: Extreme Event Detection

Evaluate whether imputation preserves the ability to detect extreme weather events. Task: Identify heat waves, cold snaps, or high wind events from imputed data. Data source: Define events using standard meteorological criteria. Relevance: Critical for climate applications where extremes matter more than means.

### 8.4 Recommendation

Start with Option B (degree days) as it requires no additional data and directly tests temperature imputation quality. Add Option C (extreme events) for the evaluation chapter. Option A can be explored if time permits and adds significant value.

---

## 9. Development Phases

### Phase 1: Foundation (Weeks 1-3)

**Objectives**: Establish data pipeline, implement classical baselines, set up evaluation framework.

**Tasks**:
1. Set up repository structure with Pydantic configs
2. Create Marimo notebook for station exploration
3. Implement GHCNh data loading utilities
4. Implement linear and spline baselines
5. Implement basic evaluation metrics (RMSE, MAE)
6. Create initial train/val/test splits
7. Set up W&B project

**Deliverables**:
- Interactive station filtering notebook with visualizations
- Working classical baselines producing reasonable results
- Documented data pipeline

**Compute**: Local only (3070 Ti sufficient)

### Phase 2: SAITS Implementation (Weeks 4-6)

**Objectives**: Implement and validate attention-based imputation.

**Tasks**:
1. Implement SAITS architecture with weather adaptations
2. Implement training loop with checkpointing
3. Create SkyPilot job configurations
4. Run initial training on regional subset (local)
5. Validate against classical baselines
6. Tune hyperparameters on validation set

**Deliverables**:
- Working SAITS implementation
- Comparison showing SAITS outperforms classical methods
- SkyPilot configs ready for cloud training

**Compute**: Local for development, first cloud runs ($10-20)

### Phase 3: CSDI Implementation (Weeks 7-9)

**Objectives**: Implement probabilistic imputation, add uncertainty quantification.

**Tasks**:
1. Implement CSDI architecture
2. Add probabilistic metrics (CRPS, calibration)
3. Compare SAITS vs CSDI on regional data
4. Implement MICE for probabilistic baseline comparison
5. Run full-scale training for SAITS (cloud)

**Deliverables**:
- Working CSDI implementation
- Probabilistic evaluation framework
- SAITS results on full dataset

**Compute**: Cloud training begins in earnest ($40-60)

### Phase 4: Full Evaluation (Weeks 10-12)

**Objectives**: Comprehensive evaluation, downstream validation, documentation.

**Tasks**:
1. Run CSDI on full dataset (cloud)
2. Complete stratified evaluation (gap length, season, extremes)
3. Statistical significance testing
4. Implement downstream validation (degree days)
5. Create results analysis notebook
6. Write up findings in publication-ready format

**Deliverables**:
- Complete results tables with confidence intervals
- Figures suitable for publication
- Draft methodology and results sections
- Archived experiment logs

**Compute**: Final cloud runs ($40-60)

---

## 10. Documentation and Publication Preparation

### 10.1 Logging Standards

Every experiment logs a unique experiment ID with timestamp, full Pydantic configuration as JSON, git commit hash, all hyperparameters to W&B, and training curves and evaluation metrics. This ensures any result can be reproduced.

### 10.2 Notebook Archives

Marimo notebooks are exported to HTML at key milestones and stored in `notebooks/exports/` with date prefixes. These serve as supplementary material for any publication.

### 10.3 Figure Generation

All figures for potential publication are generated programmatically from saved results. The results analysis notebook (`04_results_analysis.py`) produces publication-quality figures using matplotlib with consistent styling. Figures are saved in both PNG (for drafts) and PDF (for publication) formats.

### 10.4 Writing Preparation

The repository includes a `docs/` directory (not shown in structure above) for accumulating written content: methodology descriptions matching the code, results summaries updated as experiments complete, and related work notes from literature review. This content can be assembled into a paper when results are complete.

---

## 11. Risk Mitigation

### 11.1 Budget Overrun

**Risk**: Training takes longer than estimated, exceeding $300 budget.

**Mitigation**: Start with aggressive early stopping (patience=5 epochs). Use learning rate scheduling to accelerate convergence. Monitor costs in real-time via cloud dashboards. Have a fallback plan: report regional results if full-scale training exceeds budget.

### 11.2 Spot Instance Availability

**Risk**: Spot instances unavailable during training runs.

**Mitigation**: SkyPilot's multi-cloud support allows failover to alternative providers. Configure jobs to accept multiple GPU types (T4 or A10). Schedule training during off-peak hours (nights, weekends).

### 11.3 Data Quality Issues

**Risk**: GHCNh data has unexpected quality issues affecting results.

**Mitigation**: Phase 1 exploration notebook will identify issues early. Quality flag filtering provides a safety net. Document any data issues discovered for transparency.

### 11.4 Scope Creep

**Risk**: Adding features delays completion beyond 3-month window.

**Mitigation**: Strict phase gating—complete each phase before starting the next. Tier 2 variables and spatial modeling are explicitly deferred to future work. The minimum viable result is SAITS vs classical methods on Tier 1 variables.

---

## 12. Dependencies

```toml
[project]
name = "weather-imputation"
requires-python = ">=3.10"
dependencies = [
    # Core ML
    "torch>=2.0",
    
    # Data handling
    "polars>=1.0",
    "pyarrow>=14.0",
    "numpy>=1.24",
    "scipy>=1.11",
    
    # Classical methods
    "scikit-learn>=1.3",
    
    # Configuration
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    
    # Notebooks
    "marimo>=0.8",
    
    # Logging
    "wandb>=0.16",
    
    # Cloud
    "skypilot[aws,gcp]>=0.6",
    
    # Evaluation
    "properscoring>=0.1",
    
    # Utilities
    "tqdm>=4.66",
    "rich>=13.0",
    "typer>=0.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.1",
    "mypy>=1.7",
    "matplotlib>=3.8",
    "seaborn>=0.13",
]
```

---

## 13. Next Steps

Immediate actions to begin Phase 1:

1. **Initialize repository** with the directory structure and pyproject.toml
2. **Download station list** from GHCNh and create initial filtering notebook
3. **Implement GHCNh loader** that reads parquet files for a single station
4. **Create exploration notebook** for station filtering criteria
5. **Implement linear baseline** as the simplest test of the evaluation pipeline
6. **Set up W&B** project for experiment tracking
7. **Create SkyPilot account** and verify cloud provider credentials

The station exploration notebook is the critical first deliverable, as it will determine the final dataset scope and inform all subsequent work.
