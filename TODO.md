# Weather Imputation Research - Task Backlog

Generated from: `docs/weather_imputation_dev_plan_v2.md`

## Completion Criteria

- [ ] Phase 1 complete: Data pipeline, classical baselines, evaluation framework working
- [ ] Phase 2 complete: SAITS implementation validated against classical baselines
- [ ] Phase 3 complete: CSDI implementation with probabilistic metrics
- [ ] Phase 4 complete: Full evaluation, downstream validation, publication-ready results
- [ ] All code has passing tests
- [ ] W&B experiment tracking operational
- [ ] SkyPilot cloud training configurations validated
- [ ] Documentation complete and notebooks archived

---

## Assumptions

1. GHCNh parquet files are already being downloaded via existing `ghcnh_downloader.py`
2. Metadata computation infrastructure exists and is functional
3. Focus is on Tier 1 variables initially (6 core continuous variables)
4. Using Strategy D (simulate real missing) for train/val/test splits initially
5. PyTorch 2.0+ with CUDA available for local development
6. W&B account will be set up before training tasks begin

---

## PHASE 1: Foundation (Weeks 1-3)

### Configuration & Infrastructure

#### TASK-001: Create base Pydantic configuration classes
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_config.py::test_base_config -v` passes
- **Context:** `src/weather_imputation/config/base.py`, `tests/test_config.py`
- **Notes:** Completed 2026-01-25. Created BaseConfig and ExperimentConfig classes with YAML/JSON serialization support. All 14 tests passing.

#### TASK-002: Create data configuration classes
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_config.py::test_data_config -v` passes
- **Context:** `src/weather_imputation/config/data.py`, `tests/test_config.py`
- **Notes:** Completed 2026-01-25. Created StationFilterConfig, NormalizationConfig, MaskingConfig, SplitConfig, and DataConfig with comprehensive validation. All 17 tests passing.

#### TASK-003: Create model configuration classes
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_config.py::test_model_config -v` passes
- **Context:** `src/weather_imputation/config/model.py`, `tests/test_config.py`
- **Notes:** Completed 2026-01-25. Created 6 model configuration classes: ModelConfig (base), LinearInterpolationConfig, SplineInterpolationConfig, MICEConfig, SAITSConfig, CSDIConfig. Added 14 comprehensive tests, all passing. Implemented cross-field validation for d_model/n_heads divisibility and beta_start/beta_end ordering.

#### TASK-004: Create training configuration classes
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_config.py::test_training_config -v` passes
- **Context:** `src/weather_imputation/config/training.py`, `tests/test_config.py`
- **Notes:** Completed 2026-01-25. Created 5 configuration classes: OptimizerConfig, SchedulerConfig, EarlyStoppingConfig, CheckpointConfig, and TrainingConfig. Added 17 comprehensive tests, all passing. Implements NFR-004, NFR-006, NFR-009 requirements from SPEC.md.

#### TASK-005: Create evaluation configuration classes
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_config.py::test_evaluation_config -v` passes
- **Context:** `src/weather_imputation/config/evaluation.py`, `tests/test_config.py`
- **Notes:** Completed 2026-01-25. Created 5 evaluation configuration classes: MetricConfig, StratificationConfig, StatisticalTestConfig, DownstreamValidationConfig, and EvaluationConfig. Added 14 comprehensive tests, all passing. Implements FR-011 through FR-016 from SPEC.md.

### Data Pipeline

#### TASK-006: Create GHCNh variable extraction utilities
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_ghcnh_loader.py::test_extract_tier1_variables -v` passes
- **Context:** `src/weather_imputation/data/ghcnh_loader.py`, `tests/test_ghcnh_loader.py`
- **Notes:** Completed 2026-01-25. Created `extract_tier1_variables()` function that extracts Tier 1 weather variables (6 core variables) with all their attributes (value, Quality_Code, Measurement_Code, Report_Type_Code, Source_Code, units). Added 11 comprehensive tests covering default extraction, subsets, missing columns, null values, and data preservation. All tests passing. Implementation leverages existing TIER1_VARIABLES and VARIABLE_SUFFIXES constants.

#### TASK-007: Implement quality flag filtering
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_ghcnh_loader.py::test_quality_filtering -v` passes
- **Context:** `src/weather_imputation/data/ghcnh_loader.py`, `tests/test_ghcnh_loader.py`
- **Notes:** Completed 2026-01-25. Created `filter_by_quality_flags()` function that filters observations by quality control flags. By default excludes erroneous values (QC codes 3, 7), with optional suspect exclusion (QC codes 2, 6). Sets poor-quality values to null while preserving Quality_Code columns for reference. Added 13 comprehensive tests covering default behavior, suspect exclusion, subset filtering, missing columns, null handling, and edge cases. All tests passing. Implementation based on GHCNh documentation Section VI quality code definitions.

#### TASK-008: Create masking strategy for MCAR (Missing Completely At Random)
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_masking.py::test_mcar_masking -v` passes
- **Context:** `src/weather_imputation/data/masking.py`, `tests/test_masking.py`
- **Notes:** Completed 2026-01-25. Implemented MCAR masking with gap-based strategy, 17 tests passing. Added PyTorch and NumPy as dependencies.

#### TASK-009: Create masking strategy for MAR (Missing At Random)
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_masking.py::test_mar_masking -v` passes
- **Context:** `src/weather_imputation/data/masking.py`, `tests/test_masking.py`
- **Notes:** Completed 2026-01-25. Implemented MAR masking with bias towards extreme values (3x probability). All 15 tests passing.

#### TASK-010: Create masking strategy for MNAR (Missing Not At Random)
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_masking.py::test_mnar_masking -v` passes
- **Context:** `src/weather_imputation/data/masking.py`, `tests/test_masking.py`
- **Notes:** Completed 2026-01-25. Implemented MNAR masking where missingness depends on unobserved values themselves (extreme values 5x more likely to be missing by default). Added 18 comprehensive tests covering bias verification, parameter validation, edge cases, and MNAR vs MAR comparison. All 50 masking tests passing. CONFIDENCE: KNOWN - Implements FR-005 from SPEC.md (MNAR gap generation).

#### TASK-011: Create realistic gap generator based on observed patterns
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_masking.py::test_realistic_gap_generator -v` passes
- **Context:** `src/weather_imputation/data/masking.py`, `tests/test_masking.py`
- **Notes:** Completed 2026-01-25. Implemented `apply_realistic_mask()` with empirical and log-normal gap distributions. Gap distribution based on GHCNh metadata analysis: 50% short (1-6h), 40% medium (6-72h), 10% long (72-336h) gaps. All 13 new tests passing, 63 total masking tests passing. CONFIDENCE: INFERRED - Gap distribution parameters based on empirical GHCNh data analysis.

#### TASK-012: Implement per-variable normalization utilities
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_normalization.py::test_normalize_variables -v` passes
- **Context:** `src/weather_imputation/data/normalization.py`, `tests/test_normalization.py`
- **Notes:** Completed 2026-01-25. Created Normalizer class with zscore, minmax, and none normalization methods. Handles missing values gracefully (only normalizes observed values). Added 23 comprehensive tests covering fit/transform/inverse_transform, edge cases (constant variables, mostly missing data), and reproducibility. All tests passing. CONFIDENCE: KNOWN - Implements FR-007 from SPEC.md (per-variable normalization).

#### TASK-013: Implement Strategy D train/val/test splitting
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_splits.py::test_strategy_d_split -v` passes
- **Context:** `src/weather_imputation/data/splits.py`, `tests/test_splits.py`
- **Notes:** Completed 2026-01-25. Implemented all 4 splitting strategies (spatial, temporal, hybrid, simulated). Strategy D (simulated) returns all stations for all splits, with actual differentiation happening via synthetic masking during training. Created 39 comprehensive tests, all passing. Includes temporal mask utilities for splitting time series data. CONFIDENCE: KNOWN - Implements FR-006 from SPEC.md and matches SAITS/CSDI training methodology.

#### TASK-014: Create PyTorch Dataset for time series windows
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_dataset.py::test_timeseries_dataset -v` passes
- **Context:** `src/weather_imputation/data/dataset.py`, `tests/test_dataset.py`
- **Notes:** Completed 2026-01-26. Created TimeSeriesImputationDataset with sliding window extraction, synthetic masking support, and station metadata conditioning. Added 25 comprehensive tests, all passing. Ruff checks passing.

#### TASK-015: Implement DataLoader with collation function
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_dataset.py::test_dataloader_collation -v` passes
- **Context:** `src/weather_imputation/data/dataset.py`, `tests/test_dataset.py`
- **Notes:** Completed 2026-01-26. Custom collation function not needed - PyTorch's default collate_fn correctly handles dictionary outputs with tensors. Added comprehensive test covering all DataLoader scenarios: basic batching, station features, synthetic masking, shuffling, batch sizes, and drop_last. All 26 dataset tests passing.

### Classical Baselines

#### TASK-016: Implement BaseImputer protocol
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_base_imputer.py::test_protocol_compliance -v` passes
- **Context:** `src/weather_imputation/models/base.py`, `tests/test_base_imputer.py`
- **Notes:** Completed 2026-01-26. Created Imputer Protocol and BaseImputer base class with 28 comprehensive tests (all passing). Implements SPEC.md section 6.2 API contract using PyTorch tensors.

#### TASK-017: Implement linear interpolation baseline
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_models.py::test_linear_interpolation -v` passes
- **Context:** `src/weather_imputation/models/classical/linear.py`, `tests/test_models.py`
- **Notes:** Completed 2026-01-26. Implemented LinearInterpolationImputer with 25 comprehensive tests (all passing). Supports max_gap_length parameter to limit interpolation to short gaps. Uses PyTorch tensors conforming to new Imputer protocol. Handles edge cases: leading/trailing missing values, all observed/missing, multiple variables/samples. Ruff checks passing.

#### TASK-018: Implement Akima spline interpolation
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_models.py::test_akima_spline -v` passes
- **Context:** `src/weather_imputation/models/classical/spline.py`, `tests/test_models.py`
- **Notes:** Completed 2026-01-26. Implemented AkimaSplineImputer using scipy.interpolate.Akima1DInterpolator. Created 24 comprehensive tests (all passing). Added scipy>=1.10.0 dependency. Ruff checks passing. Uses PyTorch tensors conforming to Imputer protocol. Includes linear fallback for edge cases.

#### TASK-019: Implement MICE (Multiple Imputation by Chained Equations)
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_models.py::test_mice_imputation -v` passes
- **Context:** `src/weather_imputation/models/classical/mice.py`, `tests/test_models.py`
- **Notes:** Completed 2026-01-26. Implemented MICEImputer with sklearn IterativeImputer. Supports 3 predictor methods (bayesian_ridge, random_forest, linear). Generates multiple imputations for uncertainty quantification. All 19 tests passing. Added scikit-learn>=1.3.0 dependency.

### Evaluation Framework

#### TASK-020: Implement RMSE metric
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_metrics.py::test_rmse -v` passes
- **Context:** `src/weather_imputation/evaluation/metrics.py`, `tests/test_metrics.py`
- **Notes:** Completed 2026-01-26. Implemented all point metrics (RMSE, MAE, MSE, Bias, R²) using PyTorch tensors. 31 tests passing. Ruff checks passing.

#### TASK-021: Implement MAE metric
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_metrics.py::test_mae -v` passes
- **Context:** `src/weather_imputation/evaluation/metrics.py`, `tests/test_metrics.py`
- **Notes:** Completed 2026-01-26 alongside TASK-020. Implemented as part of comprehensive metrics module.

#### TASK-022: Implement Bias metric
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_metrics.py::test_bias -v` passes
- **Context:** `src/weather_imputation/evaluation/metrics.py`, `tests/test_metrics.py`
- **Notes:** Completed 2026-01-26 alongside TASK-020. Implemented as part of comprehensive metrics module.

#### TASK-023: Implement R² score metric
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_metrics.py::test_r2_score -v` passes
- **Context:** `src/weather_imputation/evaluation/metrics.py`, `tests/test_metrics.py`
- **Notes:** Completed 2026-01-26 alongside TASK-020. Implemented as part of comprehensive metrics module.

#### TASK-024: Implement gap-length stratified evaluation
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_stratified.py::test_gap_length_stratification -v` passes
- **Context:** `src/weather_imputation/evaluation/stratified.py`, `tests/test_stratified.py`
- **Notes:** Completed 2026-01-26. Completely rewrote stratified.py to use PyTorch tensors. Implemented 4 stratification types (gap length, variable, extremes, season) + convenience function. All 24 tests passing. Ruff checks passing. CONFIDENCE: KNOWN - Implements FR-013 from SPEC.md (stratified analysis).

#### TASK-025: Implement seasonal stratified evaluation
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_stratified.py::test_seasonal_stratification -v` passes
- **Context:** `src/weather_imputation/evaluation/stratified.py`, `tests/test_stratified.py`
- **Notes:** Completed 2026-01-26 as part of TASK-024. Implemented `stratify_by_season()` with meteorological season grouping. 3 tests passing.

#### TASK-026: Implement per-variable evaluation
- **Status:** DONE
- **Done When:** `uv run pytest tests/test_stratified.py::test_per_variable_evaluation -v` passes
- **Context:** `src/weather_imputation/evaluation/stratified.py`, `tests/test_stratified.py`
- **Notes:** Completed 2026-01-26 as part of TASK-024. Implemented `stratify_by_variable()` for per-variable metrics. 4 tests passing.

### Marimo Notebooks

#### TASK-027: Create station exploration notebook with filtering UI
- **Status:** DONE
- **Done When:** `marimo edit notebooks/01_station_exploration.py` opens without errors and displays interactive controls
- **Context:** `notebooks/01_station_exploration.py`, `src/weather_imputation/data/ghcnh_loader.py`
- **Notes:** Completed 2026-01-26. Notebook existed from v0.2.4 but had variable name conflicts and linting issues. Fixed marimo validation errors (variable 'output' in multiple cells), removed unused variables, fixed import sorting, resolved line length issues. All marimo check and ruff validation passing. Notebook includes: filtering controls (temp completeness, min years, min observations, coord validation), distribution histograms, geographic map with folium, K-means clustering with configurable features, cluster visualizations, and CSV export functionality.

#### TASK-028: Create gap analysis notebook
- **Status:** DONE
- **Done When:** `marimo edit notebooks/02_gap_analysis.py` opens without errors and displays gap distributions
- **Context:** `notebooks/02_gap_analysis.py`, `src/weather_imputation/data/masking.py`
- **Notes:** Completed 2026-01-26. Created interactive notebook with:
  - Station selection UI for exploring different locations
  - Natural gap analysis (identifies and characterizes gaps in real data)
  - Gap length distribution visualization with log scale
  - Gap categorization (short/medium/long)
  - Synthetic masking comparison (MCAR/MAR/MNAR/Realistic)
  - Side-by-side comparison of natural vs synthetic gap patterns
  - Summary of insights and recommendations for model training
  All marimo validation and ruff checks passing.

### Utilities

#### TASK-029: Implement circular statistics for wind direction
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_circular.py::test_wind_direction_encoding -v` passes
- **Context:** `src/weather_imputation/utils/circular.py`, `tests/test_circular.py`
- **Notes:**

#### TASK-030: Implement reproducibility utilities (seed management)
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_reproducibility.py::test_seed_everything -v` passes
- **Context:** `src/weather_imputation/utils/reproducibility.py`, `tests/test_reproducibility.py`
- **Notes:**

### Integration & Scripts

#### TASK-031: Create preprocessing script
- **Status:** TODO
- **Done When:** `uv run python src/scripts/preprocess.py --help` displays usage without errors
- **Context:** `src/scripts/preprocess.py`, `src/weather_imputation/data/`
- **Notes:**

#### TASK-032: Create evaluation script CLI
- **Status:** TODO
- **Done When:** `uv run python src/scripts/evaluate.py --help` displays usage without errors
- **Context:** `src/scripts/evaluate.py`, `src/weather_imputation/evaluation/`
- **Notes:**

#### TASK-033: Set up W&B project and integration
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_logging.py::test_wandb_initialization -v` passes
- **Context:** `src/weather_imputation/utils/logging.py`, `tests/test_logging.py`
- **Notes:**

#### TASK-034: Create default YAML configurations for classical methods
- **Status:** TODO
- **Done When:** All files in `configs/model/` validate against Pydantic schemas
- **Context:** `configs/model/linear.yaml`, `configs/model/spline.yaml`, `configs/model/mice.yaml`
- **Notes:**

#### TASK-035: Run end-to-end test with classical baselines
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_e2e.py::test_classical_baseline_pipeline -v` passes
- **Context:** `tests/test_e2e.py`, entire `src/weather_imputation/` package
- **Notes:**

---

## PHASE 2: SAITS Implementation (Weeks 4-6)

### SAITS Architecture

#### TASK-036: Implement position encoding (learnable + sinusoidal)
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_saits.py::test_position_encoding -v` passes
- **Context:** `src/weather_imputation/models/attention/saits.py`, `tests/test_saits.py`
- **Notes:**

#### TASK-037: Implement diagonally-masked self-attention (DMSA) block
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_saits.py::test_dmsa_block -v` passes
- **Context:** `src/weather_imputation/models/attention/saits.py`, `tests/test_saits.py`
- **Notes:**

#### TASK-038: Implement SAITS encoder architecture
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_saits.py::test_saits_encoder -v` passes
- **Context:** `src/weather_imputation/models/attention/saits.py`, `tests/test_saits.py`
- **Notes:**

#### TASK-039: Implement MIT (Masked Imputation Task) loss
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_saits.py::test_mit_loss -v` passes
- **Context:** `src/weather_imputation/models/attention/saits.py`, `tests/test_saits.py`
- **Notes:**

#### TASK-040: Implement ORT (Observed Reconstruction Task) loss
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_saits.py::test_ort_loss -v` passes
- **Context:** `src/weather_imputation/models/attention/saits.py`, `tests/test_saits.py`
- **Notes:**

#### TASK-041: Implement complete SAITS model with joint loss
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_saits.py::test_saits_forward_pass -v` passes
- **Context:** `src/weather_imputation/models/attention/saits.py`, `tests/test_saits.py`
- **Notes:**

#### TASK-042: Add wind direction encoding to SAITS
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_saits.py::test_wind_direction_encoding -v` passes
- **Context:** `src/weather_imputation/models/attention/saits.py`, `tests/test_saits.py`
- **Notes:**

#### TASK-043: Add station metadata conditioning (optional)
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_saits.py::test_metadata_conditioning -v` passes
- **Context:** `src/weather_imputation/models/attention/saits.py`, `tests/test_saits.py`
- **Notes:**

### Training Infrastructure

#### TASK-044: Implement base Trainer class
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_trainer.py::test_trainer_initialization -v` passes
- **Context:** `src/weather_imputation/training/trainer.py`, `tests/test_trainer.py`
- **Notes:**

#### TASK-045: Implement training loop with mixed precision
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_trainer.py::test_training_loop -v` passes
- **Context:** `src/weather_imputation/training/trainer.py`, `tests/test_trainer.py`
- **Notes:**

#### TASK-046: Implement validation loop
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_trainer.py::test_validation_loop -v` passes
- **Context:** `src/weather_imputation/training/trainer.py`, `tests/test_trainer.py`
- **Notes:**

#### TASK-047: Implement checkpoint callback
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_callbacks.py::test_checkpoint_callback -v` passes
- **Context:** `src/weather_imputation/training/callbacks.py`, `tests/test_callbacks.py`
- **Notes:**

#### TASK-048: Implement W&B logging callback
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_callbacks.py::test_wandb_callback -v` passes
- **Context:** `src/weather_imputation/training/callbacks.py`, `tests/test_callbacks.py`
- **Notes:**

#### TASK-049: Implement early stopping callback
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_callbacks.py::test_early_stopping -v` passes
- **Context:** `src/weather_imputation/training/callbacks.py`, `tests/test_callbacks.py`
- **Notes:**

#### TASK-050: Implement SkyPilot-compatible checkpoint manager
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_checkpoint.py::test_checkpoint_save_load -v` passes
- **Context:** `src/weather_imputation/training/checkpoint.py`, `tests/test_checkpoint.py`
- **Notes:**

#### TASK-051: Implement RNG state saving and restoration
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_checkpoint.py::test_rng_state_restoration -v` passes
- **Context:** `src/weather_imputation/training/checkpoint.py`, `tests/test_checkpoint.py`
- **Notes:**

### Scripts & Configurations

#### TASK-052: Create training script CLI
- **Status:** TODO
- **Done When:** `uv run python src/scripts/train.py --help` displays usage without errors
- **Context:** `src/scripts/train.py`, `src/weather_imputation/training/`
- **Notes:**

#### TASK-053: Create SAITS YAML configuration
- **Status:** TODO
- **Done When:** SAITS config loads and validates via Pydantic
- **Context:** `configs/model/saits.yaml`, `src/weather_imputation/config/model.py`
- **Notes:**

#### TASK-054: Create baseline comparison experiment config
- **Status:** TODO
- **Done When:** Experiment config loads and validates via Pydantic
- **Context:** `configs/experiment/baseline_comparison.yaml`
- **Notes:**

#### TASK-055: Create SkyPilot training job template
- **Status:** TODO
- **Done When:** `sky launch --dry-run sky/train.yaml` succeeds
- **Context:** `sky/train.yaml`, `sky/setup.sh`
- **Notes:**

### Local Training & Validation

#### TASK-056: Train SAITS on small regional subset (local)
- **Status:** TODO
- **Done When:** Training completes without errors and checkpoints are saved
- **Context:** `src/scripts/train.py`, `configs/model/saits.yaml`
- **Notes:**

#### TASK-057: Evaluate SAITS vs classical baselines on regional data
- **Status:** TODO
- **Done When:** `uv run python src/scripts/evaluate.py --models linear,spline,mice,saits` completes and shows SAITS outperforms baselines
- **Context:** `src/scripts/evaluate.py`, `src/weather_imputation/evaluation/`
- **Notes:**

#### TASK-058: Create model development notebook for SAITS
- **Status:** TODO
- **Done When:** `marimo edit notebooks/03_model_development.py` opens and displays SAITS training curves
- **Context:** `notebooks/03_model_development.py`
- **Notes:**

#### TASK-059: Perform hyperparameter search on validation set
- **Status:** TODO
- **Done When:** Hyperparameter sweep completes in W&B with convergence
- **Context:** `src/scripts/train.py`, W&B sweeps
- **Notes:**

---

## PHASE 3: CSDI Implementation (Weeks 7-9)

### CSDI Architecture

#### TASK-060: Implement noise schedule (cosine)
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_csdi.py::test_cosine_noise_schedule -v` passes
- **Context:** `src/weather_imputation/models/diffusion/csdi.py`, `tests/test_csdi.py`
- **Notes:**

#### TASK-061: Implement forward diffusion process
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_csdi.py::test_forward_diffusion -v` passes
- **Context:** `src/weather_imputation/models/diffusion/csdi.py`, `tests/test_csdi.py`
- **Notes:**

#### TASK-062: Implement transformer-based denoising network
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_csdi.py::test_denoising_network -v` passes
- **Context:** `src/weather_imputation/models/diffusion/csdi.py`, `tests/test_csdi.py`
- **Notes:**

#### TASK-063: Implement conditioning mechanism
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_csdi.py::test_conditioning -v` passes
- **Context:** `src/weather_imputation/models/diffusion/csdi.py`, `tests/test_csdi.py`
- **Notes:**

#### TASK-064: Implement reverse diffusion sampling
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_csdi.py::test_reverse_sampling -v` passes
- **Context:** `src/weather_imputation/models/diffusion/csdi.py`, `tests/test_csdi.py`
- **Notes:**

#### TASK-065: Implement multi-sample generation for uncertainty
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_csdi.py::test_multi_sample_generation -v` passes
- **Context:** `src/weather_imputation/models/diffusion/csdi.py`, `tests/test_csdi.py`
- **Notes:**

#### TASK-066: Implement self-supervised masking for training
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_csdi.py::test_self_supervised_masking -v` passes
- **Context:** `src/weather_imputation/models/diffusion/csdi.py`, `tests/test_csdi.py`
- **Notes:**

### Probabilistic Evaluation

#### TASK-067: Implement CRPS (Continuous Ranked Probability Score)
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_metrics.py::test_crps -v` passes
- **Context:** `src/weather_imputation/evaluation/metrics.py`, `tests/test_metrics.py`
- **Notes:**

#### TASK-068: Implement prediction interval calibration
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_metrics.py::test_calibration -v` passes
- **Context:** `src/weather_imputation/evaluation/metrics.py`, `tests/test_metrics.py`
- **Notes:**

#### TASK-069: Implement coverage diagnostics
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_metrics.py::test_coverage_diagnostics -v` passes
- **Context:** `src/weather_imputation/evaluation/metrics.py`, `tests/test_metrics.py`
- **Notes:**

### Scripts & Configurations

#### TASK-070: Create CSDI YAML configuration
- **Status:** TODO
- **Done When:** CSDI config loads and validates via Pydantic
- **Context:** `configs/model/csdi.yaml`, `src/weather_imputation/config/model.py`
- **Notes:**

#### TASK-071: Update evaluation script for probabilistic metrics
- **Status:** TODO
- **Done When:** `uv run python src/scripts/evaluate.py --probabilistic` runs without errors
- **Context:** `src/scripts/evaluate.py`, `src/weather_imputation/evaluation/metrics.py`
- **Notes:**

### Training & Comparison

#### TASK-072: Train CSDI on regional subset (local)
- **Status:** TODO
- **Done When:** CSDI training completes and checkpoints are saved
- **Context:** `src/scripts/train.py`, `configs/model/csdi.yaml`
- **Notes:**

#### TASK-073: Compare SAITS vs CSDI on regional data
- **Status:** TODO
- **Done When:** Evaluation shows CSDI provides calibrated uncertainty estimates
- **Context:** `src/scripts/evaluate.py`
- **Notes:**

#### TASK-074: Run full-scale SAITS training on cloud
- **Status:** TODO
- **Done When:** `sky launch sky/train.yaml` completes successfully and results logged to W&B
- **Context:** `sky/train.yaml`, `src/scripts/train.py`
- **Notes:**

---

## PHASE 4: Full Evaluation (Weeks 10-12)

### Cloud Training

#### TASK-075: Run full-scale CSDI training on cloud
- **Status:** TODO
- **Done When:** CSDI cloud training completes and results logged to W&B
- **Context:** `sky/train.yaml`, `src/scripts/train.py`
- **Notes:**

#### TASK-076: Create SkyPilot evaluation job template
- **Status:** TODO
- **Done When:** `sky launch --dry-run sky/evaluate.yaml` succeeds
- **Context:** `sky/evaluate.yaml`
- **Notes:**

### Statistical Analysis

#### TASK-077: Implement paired Wilcoxon signed-rank test
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_statistical.py::test_wilcoxon_test -v` passes
- **Context:** `src/weather_imputation/evaluation/statistical.py`, `tests/test_statistical.py`
- **Notes:**

#### TASK-078: Implement Bonferroni correction
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_statistical.py::test_bonferroni_correction -v` passes
- **Context:** `src/weather_imputation/evaluation/statistical.py`, `tests/test_statistical.py`
- **Notes:**

#### TASK-079: Implement Cohen's d effect size
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_statistical.py::test_cohens_d -v` passes
- **Context:** `src/weather_imputation/evaluation/statistical.py`, `tests/test_statistical.py`
- **Notes:**

#### TASK-080: Implement bootstrap confidence intervals
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_statistical.py::test_bootstrap_ci -v` passes
- **Context:** `src/weather_imputation/evaluation/statistical.py`, `tests/test_statistical.py`
- **Notes:**

### Stratified Evaluation

#### TASK-081: Implement extreme value evaluation (5th/95th percentiles)
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_stratified.py::test_extreme_value_evaluation -v` passes
- **Context:** `src/weather_imputation/evaluation/stratified.py`, `tests/test_stratified.py`
- **Notes:**

#### TASK-082: Run complete stratified evaluation (gap length, season, variable, extremes)
- **Status:** TODO
- **Done When:** `uv run python src/scripts/evaluate.py --stratified` completes and generates all result tables
- **Context:** `src/scripts/evaluate.py`, `src/weather_imputation/evaluation/stratified.py`
- **Notes:**

### Downstream Validation

#### TASK-083: Implement degree day computation utilities
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_degree_days.py::test_growing_degree_days -v` passes
- **Context:** `src/weather_imputation/evaluation/degree_days.py`, `tests/test_degree_days.py`
- **Notes:**

#### TASK-084: Implement heating/cooling degree days
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_degree_days.py::test_heating_cooling_degree_days -v` passes
- **Context:** `src/weather_imputation/evaluation/degree_days.py`, `tests/test_degree_days.py`
- **Notes:**

#### TASK-085: Evaluate degree day preservation across imputation methods
- **Status:** TODO
- **Done When:** Downstream validation results show imputation preserves degree day statistics
- **Context:** `src/scripts/evaluate.py`, `src/weather_imputation/evaluation/degree_days.py`
- **Notes:**

#### TASK-086: Implement extreme event detection utilities
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_extreme_events.py::test_heat_wave_detection -v` passes
- **Context:** `src/weather_imputation/evaluation/extreme_events.py`, `tests/test_extreme_events.py`
- **Notes:**

#### TASK-087: Evaluate extreme event preservation across imputation methods
- **Status:** TODO
- **Done When:** Extreme event analysis shows imputation maintains event detection capability
- **Context:** `src/scripts/evaluate.py`, `src/weather_imputation/evaluation/extreme_events.py`
- **Notes:**

### Results & Documentation

#### TASK-088: Create results analysis notebook
- **Status:** TODO
- **Done When:** `marimo edit notebooks/04_results_analysis.py` opens and displays all publication figures
- **Context:** `notebooks/04_results_analysis.py`
- **Notes:**

#### TASK-089: Generate publication-quality figures
- **Status:** TODO
- **Done When:** All figures saved to `notebooks/exports/figures/` in both PNG and PDF formats
- **Context:** `notebooks/04_results_analysis.py`
- **Notes:**

#### TASK-090: Export all notebooks to HTML
- **Status:** TODO
- **Done When:** All HTML exports saved to `notebooks/exports/` with timestamps
- **Context:** All `notebooks/*.py` files
- **Notes:**

#### TASK-091: Create complete results tables with confidence intervals
- **Status:** TODO
- **Done When:** Results tables generated in LaTeX and CSV formats
- **Context:** `notebooks/04_results_analysis.py`, `data/results/`
- **Notes:**

#### TASK-092: Write methodology documentation
- **Status:** TODO
- **Done When:** `docs/methodology.md` contains complete description matching implementation
- **Context:** `docs/methodology.md`, entire codebase
- **Notes:**

#### TASK-093: Write results summary documentation
- **Status:** TODO
- **Done When:** `docs/results_summary.md` contains all key findings with statistical tests
- **Context:** `docs/results_summary.md`, evaluation outputs
- **Notes:**

#### TASK-094: Update README with final project overview
- **Status:** TODO
- **Done When:** `README.md` contains complete project description, setup, and results overview
- **Context:** `README.md`
- **Notes:**

#### TASK-095: Update CHANGELOG with all Phase 1-4 changes
- **Status:** TODO
- **Done When:** `CHANGELOG.md` contains dated entries for all major milestones
- **Context:** `CHANGELOG.md`
- **Notes:**

---

## Additional Integration Tasks

#### TASK-096: Implement experiment tracking manifest
- **Status:** TODO
- **Done When:** Every experiment run generates a manifest JSON with git hash, config, and timestamp
- **Context:** `src/weather_imputation/utils/logging.py`, `data/results/`
- **Notes:**

#### TASK-097: Create ablation study configurations
- **Status:** TODO
- **Done When:** Ablation configs validate and test different SAITS/CSDI components
- **Context:** `configs/experiment/ablation_study.yaml`
- **Notes:**

#### TASK-098: Implement result reproducibility check
- **Status:** TODO
- **Done When:** `uv run pytest tests/test_reproducibility.py::test_deterministic_training -v` passes
- **Context:** `tests/test_reproducibility.py`
- **Notes:**

#### TASK-099: Create regional test data configuration
- **Status:** TODO
- **Done When:** Regional config loads and filters to ~500-1000 stations
- **Context:** `configs/data/regional_test.yaml`
- **Notes:**

#### TASK-100: Create north america full dataset configuration
- **Status:** TODO
- **Done When:** North America config loads and includes all filtered stations
- **Context:** `configs/data/north_america.yaml`
- **Notes:**

---

## Total Tasks: 100

**Estimated Completion:**
- Phase 1: 35 tasks
- Phase 2: 24 tasks
- Phase 3: 15 tasks
- Phase 4: 21 tasks
- Integration: 5 tasks
