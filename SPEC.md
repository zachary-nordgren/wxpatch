# Weather Data Imputation Research Framework — Specification

> **Status:** APPROVED
> **Last Updated:** 2026-01-25
> **Author:** Synthesized from development plan v2

## 1. Overview

### 1.1 Problem Statement
Weather monitoring stations produce incomplete time series due to sensor failures, communication outages, and maintenance periods. Missing weather data limits the utility of downstream applications (energy demand forecasting, agricultural planning, climate analysis). Existing imputation methods either use oversimplified interpolation (linear, spline) that fails to capture complex spatiotemporal patterns, or require infrastructure unavailable to researchers ($10k+ cloud budgets, proprietary datasets).

This project addresses: How can researchers systematically compare classical and deep learning imputation methods on real-world hourly weather data within a $100-300 compute budget?

### 1.2 Solution Summary
A Python-based research framework for multi-variable weather imputation using the GHCNh (Global Historical Climatology Network hourly) dataset. The system implements classical baselines (linear, spline, MICE) and modern deep learning methods (SAITS attention-based, CSDI diffusion-based) with rigorous evaluation including statistical significance testing, stratified analysis, and downstream validation tasks. Marimo notebooks enable interactive exploration, Pydantic ensures type-safe configuration, and SkyPilot orchestrates cost-efficient cloud training on spot instances.

### 1.3 Success Criteria
- [ ] Successfully process 3,000-6,000 North American weather stations from GHCNh parquet files
- [ ] Implement 5 imputation methods: linear, spline, MICE, SAITS, CSDI
- [ ] Demonstrate SAITS outperforms classical baselines on RMSE/MAE by ≥15% (expected)
- [ ] Complete full-scale training within $100-300 cloud compute budget
- [ ] Generate publication-ready results with confidence intervals and significance tests
- [ ] Provide reproducible experiments (all configs versioned, W&B logs archived)

---

## 2. Requirements

### 2.1 Functional Requirements
| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-001 | Download and parse GHCNh parquet files for North American stations | MUST | ~8,787 stations, 1973-2023 |
| FR-002 | Compute station metadata (completeness, gap patterns, temporal coverage) | MUST | Enables filtering decisions |
| FR-003 | Filter stations by quality thresholds (completeness, time span, coordinates) | MUST | Target 3k-6k stations |
| FR-004 | Implement 6-variable imputation (temp, dewpoint, pressure, wind speed/dir, humidity) | MUST | Tier 1 variables from plan |
| FR-005 | Support 4 gap generation strategies (MCAR, MAR, MNAR, realistic) | MUST | For controlled evaluation |
| FR-006 | Implement train/val/test splits (4 strategies: spatial, temporal, hybrid, simulated) | MUST | Strategy D preferred |
| FR-007 | Normalize variables per-station (z-score or min-max) | MUST | Required for neural methods |
| FR-008 | Implement classical baselines: linear, spline (Akima), MICE | MUST | Comparison anchors |
| FR-009 | Implement SAITS with weather adaptations (circular wind encoding, metadata conditioning) | MUST | Primary deep learning method |
| FR-010 | Implement CSDI for probabilistic imputation | MUST | Uncertainty quantification |
| FR-011 | Compute point metrics (RMSE, MAE, bias) per variable | MUST | Core evaluation |
| FR-012 | Compute probabilistic metrics (CRPS, calibration) for CSDI | MUST | Probabilistic evaluation |
| FR-013 | Stratify results by gap length, season, variable, extremes | MUST | Detailed analysis |
| FR-014 | Statistical significance testing (Wilcoxon signed-rank, Bonferroni) | MUST | Method comparisons |
| FR-015 | Downstream validation: degree days computation | SHOULD | Tests practical utility |
| FR-016 | Downstream validation: extreme event detection | COULD | Heat waves, cold snaps |
| FR-017 | Interactive station filtering notebook (Marimo) | MUST | Exploratory analysis |
| FR-018 | Results analysis notebook with publication-quality figures | MUST | Visualization |
| FR-019 | Experiment tracking with W&B (configs, metrics, hyperparameters) | MUST | Reproducibility |
| FR-020 | SkyPilot cloud training with spot instance recovery | MUST | Cost control |

### 2.2 Non-Functional Requirements
| ID | Requirement | Target | Notes |
|----|-------------|--------|-------|
| NFR-001 | Cloud compute cost | $100-300 total | SkyPilot spot instances |
| NFR-002 | Development timeline | 12 weeks part-time | 4 phases |
| NFR-003 | Data processing throughput | Process all stations in <4 hours | Parallel metadata computation |
| NFR-004 | Training speed | SAITS regional convergence in <20 hours on T4 | Mixed precision, torch.compile |
| NFR-005 | Memory footprint | Fit training on single GPU (16GB VRAM) | Batch size tuning |
| NFR-006 | Reproducibility | Bit-exact results with same seed | RNG state checkpointing |
| NFR-007 | Code quality | Type-checked (mypy), linted (ruff) | CI enforcement |
| NFR-008 | Storage efficiency | <20GB processed data | Filtered parquet files |
| NFR-009 | Checkpointing frequency | Every 30 min + epoch boundaries | Spot preemption resilience |
| NFR-010 | Experiment logging | All hyperparameters, git hash, timestamps | Full provenance |

### 2.3 Out of Scope
- **Spatial interpolation models** (kriging, Gaussian processes) — deferred to future work
- **Tier 2 variables** (visibility, wet bulb temp, altimeter) — framework extensible but not implemented
- **Tier 3 variables** (categorical sky cover, precipitation with zero-inflation) — complex modeling requirements
- **Real-time imputation** — research focus is offline batch processing
- **Operational deployment** — API, microservices, monitoring infrastructure
- **Multi-site joint modeling** — initial implementation treats stations independently
- **Renewable energy validation (Option A)** — requires external NREL dataset integration
- **Production data pipelines** — Airflow, data versioning, drift detection

---

## 3. Architecture

### 3.1 System Context
The framework operates as a standalone research pipeline consuming public GHCNh parquet files from NOAA and producing evaluation results logged to Weights & Biases. Cloud training via SkyPilot interfaces with AWS/GCP/Azure spot instances but all artifacts are portable.

```
┌─────────────────────────────────────────────────────────────┐
│                   External Dependencies                      │
├─────────────────────────────────────────────────────────────┤
│  NOAA GHCNh Parquet Files (S3) → [Download Script]          │
│  Cloud Providers (AWS/GCP/Azure) ← [SkyPilot Orchestration] │
│  Weights & Biases API ← [Training Loop]                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Weather Imputation Framework                    │
├─────────────────────────────────────────────────────────────┤
│  [Data Pipeline] → [Training Infrastructure] → [Evaluation]  │
│       ↓                     ↓                       ↓        │
│  data/processed/     checkpoints/           results/         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Outputs                                    │
├─────────────────────────────────────────────────────────────┤
│  • Trained model checkpoints (PyTorch state_dict)           │
│  • Evaluation metrics (parquet tables, JSON)                │
│  • Publication figures (PDF/PNG)                             │
│  • W&B experiment logs (online dashboard)                   │
│  • Marimo notebook exports (HTML archives)                  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Component Design
| Component | Responsibility | Interfaces |
|-----------|---------------|------------|
| **data.ghcnh_loader** | Download/parse GHCNh parquet, filter by report type | `load_station(id, year) → DataFrame` |
| **data.metadata** | Compute completeness, gaps, temporal coverage | `compute_metadata(station_files) → StationMetadata` |
| **data.dataset** | PyTorch Dataset for windowed time series with masks | `__getitem__(idx) → (observed, mask, target)` |
| **data.masking** | Generate synthetic gaps (MCAR/MAR/MNAR) | `apply_mask(data, strategy) → masked_data` |
| **data.normalization** | Per-station z-score normalization | `fit(data) → Normalizer`, `transform()`, `inverse_transform()` |
| **data.splits** | Train/val/test splitting strategies | `split(metadata, strategy) → (train_ids, val_ids, test_ids)` |
| **models.base** | BaseImputer protocol (fit/impute/save/load) | Common interface for all methods |
| **models.classical** | Linear, spline, MICE implementations | `impute(observed, mask) → Tensor` |
| **models.attention.saits** | SAITS architecture (DMSA, MIT/ORT loss) | `forward(batch) → predictions, loss_dict` |
| **models.diffusion.csdi** | CSDI diffusion model (denoising network) | `forward(batch, timestep) → noise_pred` |
| **training.trainer** | Training loop, optimizer, scheduler, early stopping | `train(model, loaders, config) → checkpoint` |
| **training.callbacks** | Checkpointing, W&B logging, validation metrics | Hook-based event system |
| **evaluation.metrics** | RMSE, MAE, bias, CRPS, calibration | `compute_metrics(y_true, y_pred) → dict` |
| **evaluation.statistical** | Wilcoxon tests, Bonferroni correction, Cohen's d | `compare_methods(results) → significance_df` |
| **evaluation.stratified** | Group metrics by gap length, season, variable | `stratify_results(preds, metadata) → stratified_df` |
| **config.*** | Pydantic models for all configurations | Type-safe YAML/JSON loading |
| **utils.progress** | Rich progress bars for long operations | `track(iterable, description)` |

### 3.3 Data Flow
```
1. SETUP PHASE
   ghcnh_downloader.py → data/raw/ghcnh/{year}/{station}.parquet
                      → compute_metadata.py → data/processed/metadata.parquet
                      → clean_metadata.py → data/processed/metadata_cleaned.parquet

2. EXPLORATION PHASE (Marimo notebook)
   metadata_cleaned.parquet → 01_station_exploration.py
                           → User adjusts thresholds → selected_stations.json

3. PREPROCESSING PHASE
   preprocess.py:
     - Load selected_stations.json
     - For each station: load multi-year parquet → quality filter → normalize
     - Compute train/val/test splits (Strategy D: simulated masks)
     - Save: data/processed/train.parquet, val.parquet, test.parquet

4. TRAINING PHASE (local or cloud)
   train.py:
     - Load config (model + training hyperparameters)
     - Initialize Dataset, DataLoader
     - Initialize Model (SAITS or CSDI)
     - Trainer.train() → checkpoints/exp_{id}/checkpoint_epoch_N.pt
     - Log to W&B: loss curves, val metrics

5. EVALUATION PHASE
   evaluate.py:
     - Load checkpoint → model.load()
     - Load test.parquet → compute predictions
     - Compute metrics (point + probabilistic + stratified)
     - Statistical tests → results/comparison_table.csv
     - Generate figures → results/figures/

6. PUBLICATION PHASE (Marimo notebook)
   04_results_analysis.py:
     - Load all results/*.csv
     - Generate publication-quality figures (matplotlib)
     - Export notebook → notebooks/exports/results_YYYYMMDD.html
```

### 3.4 Data Model

#### Station Metadata (Parquet Schema)
| Field | Type | Constraints | Notes |
|-------|------|-------------|-------|
| `station_id` | `String` | Primary key, 11 chars | e.g., "72053023174" |
| `country_code` | `String` | 2 chars | "US", "CA", "MX" |
| `station_name` | `String` | | Human-readable |
| `latitude` | `Float64` | -90 to 90 | Decimal degrees |
| `longitude` | `Float64` | -180 to 180 | Decimal degrees |
| `elevation` | `Float64` | meters | Nullable |
| `first_observation` | `Datetime` | | Earliest valid timestamp |
| `last_observation` | `Datetime` | | Latest valid timestamp |
| `years_available` | `Int32` | ≥1 | Count of years with data |
| `total_observation_count` | `Int64` | | All records after filtering |
| `temperature_completeness_pct` | `Float64` | 0-100 | % non-null observations |
| `dew_point_completeness_pct` | `Float64` | 0-100 | |
| `sea_level_pressure_completeness_pct` | `Float64` | 0-100 | |
| `wind_speed_completeness_pct` | `Float64` | 0-100 | |
| `wind_direction_completeness_pct` | `Float64` | 0-100 | |
| `relative_humidity_completeness_pct` | `Float64` | 0-100 | |
| `temperature_stats` | `JSON` | `{mean, std, min, max, p25, p50, p75}` | Statistics dict |
| `gap_count_24h` | `Int64` | | Number of gaps >24 hours |
| `max_gap_duration_hours` | `Float64` | | Longest gap |
| `avg_observation_interval_hours` | `Float64` | | Typical sampling rate |
| `report_type_counts` | `JSON` | `{AUTO: 50000, FM-15: 10000}` | Record counts by type |
| `total_records_all_types` | `Int64` | | Before filtering |
| `records_excluded_by_filter` | `Int64` | | Excluded by report type |

#### Processed Dataset (PyTorch Tensors)
| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| `observations` | `(N, T, V)` | `float32` | N=samples, T=timesteps, V=6 variables |
| `mask` | `(N, T, V)` | `bool` | True=observed, False=missing |
| `target` | `(N, T, V)` | `float32` | Ground truth for masked values |
| `timestamps` | `(N, T)` | `int64` | Unix timestamps for temporal features |
| `station_features` | `(N, F)` | `float32` | Lat, lon, elevation (normalized) |

#### Configuration Schema (Pydantic)
```python
# config/data.py
class DataConfig(BaseModel):
    variables: list[str] = ["temperature", "dew_point_temperature", ...]
    station_filter: StationFilterConfig
    normalization_method: Literal["zscore", "minmax"] = "zscore"
    window_size: int = 168  # hours (1 week)
    masking_strategy: Literal["mcar", "mar", "mnar", "realistic"] = "realistic"

# config/model.py
class SAITSConfig(BaseModel):
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.1
    mit_weight: float = 1.0
    ort_weight: float = 1.0

# config/training.py
class TrainingConfig(BaseModel):
    batch_size: int = 32
    max_epochs: int = 100
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    checkpoint_every_n_minutes: int = 30
```

---

## 4. Technical Decisions

### 4.1 Technology Choices
| Decision | Choice | Rationale | Alternatives Considered |
|----------|--------|-----------|------------------------|
| **Language** | Python 3.10+ | Research ecosystem (PyTorch, scipy, sklearn), type hints | Julia (less mature ML), R (harder deployment) |
| **Notebook Environment** | Marimo | Reactive execution, pure Python files (git-friendly), built-in UI widgets | Jupyter (stale outputs, .ipynb merge conflicts), Observable (JS only) |
| **Configuration** | Pydantic v2 | Type validation, JSON schema generation, no boilerplate | Hydra (YAML-centric), OmegaConf (stringly-typed) |
| **Data Processing** | Polars | 5-10x faster than Pandas, lazy evaluation, Arrow-native | Pandas (slower), Dask (overkill for this scale) |
| **Deep Learning** | PyTorch 2.0+ | Research standard, `torch.compile`, SDPA | TensorFlow (verbose), JAX (steeper learning curve) |
| **Cloud Orchestration** | SkyPilot | Multi-cloud spot instances, auto-recovery, simple YAML | Ray (heavier), Kubernetes (complex), manual AWS CLI |
| **Experiment Tracking** | Weights & Biases | Free tier sufficient, good comparison UI, artifact storage | MLflow (self-hosted burden), TensorBoard (limited) |
| **Testing** | pytest | Standard, rich plugin ecosystem | unittest (verbose), nose (deprecated) |
| **Package Manager** | uv | 10-100x faster than pip, lockfile generation | pip (slow), Poetry (dependency resolution issues) |
| **Linting** | Ruff | Rust-based, 10-100x faster than flake8, auto-fix | flake8 + black + isort (separate tools) |

### 4.2 Conventions
- **Naming:**
  - `snake_case` for variables, functions, modules
  - `PascalCase` for classes, Pydantic models
  - `UPPER_SNAKE_CASE` for constants
  - Prefix private methods with `_`
- **File Structure:**
  - Feature-based organization (`data/`, `models/`, `evaluation/`)
  - Entry points in `scripts/`, importable logic in `weather_imputation/`
  - Tests mirror source structure: `tests/test_data/test_masking.py`
- **Testing:**
  - Unit tests colocated in `tests/`
  - Descriptive test names: `test_mcar_masking_preserves_marginal_distribution`
  - Fixtures in `conftest.py` for reusable test data
  - Mock external I/O (NOAA downloads, W&B API)
- **Documentation:**
  - Docstrings for all public functions (Google style)
  - Type hints mandatory (enforced by mypy --strict)
  - README.md for user-facing instructions
  - CLAUDE.md for agent context + codebase map
  - SPEC.md (this file) for implementation reference
- **Git:**
  - Branch: `migration/ghcnh-parquet-architecture`
  - Commit messages: Descriptive, include "Co-Authored-By: Claude Sonnet 4.5"
  - Always update CHANGELOG.md with dated entries
  - No secrets in repo (use .env, gitignored)

---

## 5. Error Handling Policy

### 5.1 Error Categories
| Category | Handling Strategy | User Impact |
|----------|------------------|-------------|
| **Data Download Failures** | Retry with exponential backoff (3 attempts), skip station if persistent | Log warning, continue with other stations |
| **File I/O Errors** | Check permissions/space upfront, fail fast with clear message | Show error with file path, suggest fixes |
| **Validation Errors (Config)** | Pydantic raises ValidationError with field details | Display validation errors with config path |
| **Missing Data (Station)** | Skip station if below completeness threshold | Log exclusion reason, report in summary |
| **CUDA OOM** | Catch in training loop, suggest reducing batch size, exit gracefully | Show memory stats, suggest config changes |
| **Spot Instance Preemption** | SkyPilot auto-restart from latest checkpoint | Transparent to user, logged in W&B |
| **Checkpoint Corruption** | Validate checkpoints on load, fall back to previous epoch | Warn user, resume from last valid checkpoint |
| **Metric Computation Errors** | Catch per-metric, return NaN, log warning | Partial results table with NaN for failed metrics |
| **External Service (W&B) Failures** | Retry 3x, fall back to local logging if offline | Warning message, continue training |
| **Unexpected Exceptions** | Log full traceback with context, save debug checkpoint, exit | Show concise error + log location |

### 5.2 Logging Strategy
**Principle:** Aggregate context along execution path; emit single structured log entry at operation boundaries.

**Implementation:**
- Use Python's `logging` module with structured fields (JSON formatter)
- Rich `Console` for user-facing progress/errors (not mixed with logs)
- Log levels:
  - `DEBUG`: Parameter values, loop iterations (disabled in production)
  - `INFO`: Phase completions ("Metadata computation done: 5000 stations")
  - `WARNING`: Recoverable issues (station skipped, metric failed)
  - `ERROR`: Operation failures requiring user action
  - `CRITICAL`: Unrecoverable errors (filesystem full, invalid config)

**Example:**
```python
# BAD (piecemeal)
logger.info("Starting download")
logger.info(f"Station: {station_id}")
logger.info(f"Year: {year}")

# GOOD (aggregated)
log_ctx = {"station_id": station_id, "year": year, "start_time": time.time()}
try:
    result = download_station(station_id, year)
    log_ctx.update({"outcome": "success", "size_mb": result.size})
except DownloadError as e:
    log_ctx.update({"outcome": "failure", "error": str(e)})
finally:
    log_ctx["duration_sec"] = time.time() - log_ctx["start_time"]
    logger.info("download_completed", extra=log_ctx)
```

### 5.3 Recovery Procedures
- **Partial Download Failures:** Resume from last successful station (track in `download_status.json`)
- **Training Interruption:** Auto-resume from latest checkpoint (SkyPilot handles this)
- **Corrupted Metadata:** Re-run `compute_metadata.py` for affected stations only
- **Failed Evaluation:** Recompute metrics from saved predictions (don't re-run inference)
- **Checkpoint Loss:** Training can restart from scratch if cloud storage failed (local backup recommended)

---

## 6. API Contracts

### 6.1 External APIs Consumed
| API | Purpose | Auth | Docs | Rate Limits |
|-----|---------|------|------|-------------|
| NOAA GHCNh S3 Bucket | Download parquet files | None (public) | [ghcnh_DOCUMENTATION.pdf](docs/) | None (request-pays disabled) |
| Weights & Biases API | Log experiments, artifacts | API key (env var) | [wandb.ai/docs](https://docs.wandb.ai) | Free tier: 100GB storage |
| AWS/GCP/Azure APIs | SkyPilot instance provisioning | Cloud credentials | Per-provider docs | Spot instance availability varies |

### 6.2 APIs Exposed
**Internal Python APIs (not web services):**

#### BaseImputer Protocol
```python
class Imputer(Protocol):
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Train the imputation model. No-op for classical methods."""
        ...

    def impute(self, observed: Tensor, mask: Tensor) -> Tensor:
        """
        Impute missing values.

        Args:
            observed: (N, T, V) tensor with observed values
            mask: (N, T, V) boolean tensor (True=observed, False=missing)

        Returns:
            (N, T, V) tensor with imputed values (masked positions filled)
        """
        ...

    def save(self, path: Path) -> None:
        """Save model state to disk."""
        ...

    def load(self, path: Path) -> None:
        """Load model state from disk."""
        ...
```

#### Metric Functions
```python
def compute_rmse(y_true: Tensor, y_pred: Tensor, mask: Tensor) -> float:
    """Compute RMSE on masked positions only."""
    ...

def compute_crps(y_true: Tensor, y_samples: Tensor, mask: Tensor) -> float:
    """
    Compute Continuous Ranked Probability Score.

    Args:
        y_true: (N, T, V) ground truth
        y_samples: (N, T, V, S) ensemble predictions (S samples)
        mask: (N, T, V) evaluation mask

    Returns:
        Mean CRPS across all masked positions
    """
    ...
```

#### Data Loading
```python
def load_station_data(
    station_id: str,
    year_range: tuple[int, int],
    variables: list[str],
    report_types: list[str] = ["AUTO", "FM-15"]
) -> pl.DataFrame:
    """
    Load and concatenate multi-year station data.

    Returns:
        Polars DataFrame with columns: DATE, variable_value, variable_Quality_Code
    """
    ...
```

---

## 7. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Budget Overrun (Training >$300)** | MEDIUM | HIGH | Aggressive early stopping (patience=5), monitor costs real-time, fallback to regional results |
| **Spot Instance Unavailability** | MEDIUM | MEDIUM | SkyPilot multi-cloud failover, accept multiple GPU types (T4/A10/L4), schedule off-peak |
| **GHCNh Data Quality Issues** | MEDIUM | MEDIUM | Phase 1 exploration catches issues early, quality flag filtering, document anomalies |
| **CSDI Training Instability** | HIGH | MEDIUM | Gradient clipping, careful noise schedule tuning, validate on SAITS first |
| **Scope Creep (Tier 2 Variables)** | HIGH | HIGH | Strict phase gating, defer to future work, minimum viable result = Tier 1 only |
| **Checkpoint Corruption (Preemption)** | LOW | MEDIUM | Validate checkpoints on save, keep last 3 checkpoints, cloud storage redundancy |
| **W&B Service Outage** | LOW | LOW | Local logging fallback, offline mode available |
| **Insufficient Disk Space (Local)** | MEDIUM | MEDIUM | Check free space upfront (100GB recommended), delete raw files after processing |
| **Marimo Notebook State Loss** | LOW | LOW | Pure Python files auto-save, export HTML at milestones |
| **Statistical Non-Significance** | MEDIUM | MEDIUM | Increase sample size (more stations), report effect sizes even if p>0.05 |

---

## 8. Open Questions

> Resolve these during Phase 1.

- [ ] **Data Filtering Thresholds:** What temperature completeness % maximizes station count while ensuring quality? (Exploration notebook will decide: 50-90% range)
- [ ] **Train/Val/Test Strategy:** Confirm Strategy D (simulated masks within signals) or hybrid approach? (Assumption: Strategy D preferred per plan)
- [ ] **Wind Direction Encoding:** Sin/cos components or von Mises distribution for CSDI? (Assumption: sin/cos for simplicity, revisit if performance poor)
- [ ] **SAITS Hyperparameters:** n_layers (2 vs 4), d_model (128 vs 256) — run grid search or use paper defaults? (Assumption: start with paper defaults, tune if time permits)
- [ ] **CSDI Diffusion Steps:** 50 vs 100 steps trade-off quality vs cost? (Assumption: 50 steps, evaluate quality on regional data first)
- [ ] **Downstream Validation Priority:** Degree days (Option B) only, or also extreme events (Option C)? (Recommendation: both if time, degree days minimum)
- [ ] **Metadata Cleaning Edge Cases:** How to handle stations with identical coords but different IDs? (Current: keep both, flag as duplicates)
- [ ] **Cloud Provider Choice:** AWS (familiar) vs GCP (cheaper T4s) vs Lambda Labs (cheapest but less reliable)? (Recommendation: SkyPilot auto-selects, prefer AWS for stability)

---

## 9. References

### Documentation
- **NOAA GHCNh Format:** [docs/ghcnh_DOCUMENTATION.pdf](docs/ghcnh_DOCUMENTATION.pdf)
- **Development Plan:** [docs/weather_imputation_dev_plan_v2.md](docs/weather_imputation_dev_plan_v2.md)
- **Task Backlog:** [TODO.md](TODO.md) (100 tasks across 4 phases)

### Papers
- **SAITS:** Du et al. (2023) "SAITS: Self-Attention-based Imputation for Time Series" ([arXiv](https://arxiv.org/abs/2202.08516))
- **CSDI:** Tashiro et al. (2021) "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation" ([NeurIPS](https://arxiv.org/abs/2107.03502))
- **MICE:** van Buuren & Groothuis-Oudshoorn (2011) "mice: Multivariate Imputation by Chained Equations in R" ([JSS](https://www.jstatsoft.org/article/view/v045i03))

### Tools
- **Marimo:** [marimo.io](https://marimo.io)
- **Pydantic:** [docs.pydantic.dev](https://docs.pydantic.dev/latest/)
- **SkyPilot:** [skypilot.readthedocs.io](https://skypilot.readthedocs.io)
- **Weights & Biases:** [docs.wandb.ai](https://docs.wandb.ai)

### Datasets
- **GHCNh Parquet Files:** [NOAA S3 Bucket](https://noaa-ghcn-pds.s3.amazonaws.com/index.html)
- **Station Inventory:** [NOAA FTP](ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.txt)
- **NREL NSRDB (Optional):** [nsrdb.nrel.gov](https://nsrdb.nrel.gov) (for downstream validation Option A)

---

**END OF SPECIFICATION**

*This document supersedes informal planning notes. All implementation decisions should reference this spec. Update this file when requirements change.*
