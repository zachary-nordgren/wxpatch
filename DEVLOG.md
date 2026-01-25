# Development Log

This log tracks implementation progress, decisions, and findings during development.

---

## 2026-01-25

### TASK-001: Base Pydantic Configuration Classes

**Status:** Completed

**Implementation:**
- Created `src/weather_imputation/config/base.py` with:
  - `BaseConfig`: Base Pydantic v2 model with YAML/JSON serialization
  - `ExperimentConfig`: Experiment metadata tracking (name, description, seed, tags)
- Created comprehensive test suite in `tests/test_config.py` (14 tests, all passing)
- Added PyYAML dependency to `pyproject.toml`
- Updated `src/weather_imputation/config/__init__.py` to export new classes

**Key Design Decisions:**
- Used Pydantic v2 `ConfigDict` for model configuration
- Set `extra="forbid"` to catch configuration errors early
- Enabled `validate_assignment=True` for runtime validation
- Provided both YAML and JSON serialization (YAML preferred for configs)
- Auto-create parent directories when saving config files

**Test Coverage:**
- Basic instantiation and defaults
- Validation errors (missing fields, wrong types, extra fields)
- Dictionary conversion
- YAML/JSON serialization and deserialization
- File I/O roundtrips
- Parent directory creation
- Nested configuration support
- Validation on assignment

**Files Modified:**
- `src/weather_imputation/config/base.py` (created)
- `src/weather_imputation/config/__init__.py` (updated exports)
- `tests/test_config.py` (created)
- `pyproject.toml` (added pyyaml dependency)
- `TODO.md` (marked TASK-001 as DONE)

**Confidence:** KNOWN - Standard Pydantic v2 patterns, well-tested implementation.

---

### TASK-002: Data Configuration Classes

**Status:** Completed

**Implementation:**
- Created `src/weather_imputation/config/data.py` with 5 configuration classes:
  - `StationFilterConfig`: Quality thresholds for filtering stations (completeness %, years, geographic bounds, gap patterns)
  - `NormalizationConfig`: Variable normalization settings (zscore/minmax/none, per-station/global, outlier clipping)
  - `MaskingConfig`: Synthetic gap generation (MCAR/MAR/MNAR/realistic strategies, missing ratio, gap length constraints)
  - `SplitConfig`: Train/val/test splitting (spatial/temporal/hybrid/simulated strategies, ratio validation)
  - `DataConfig`: Main configuration aggregating all above configs + Tier 1 variables, windowing, report types
- Added 17 comprehensive tests to `tests/test_config.py` (all passing)
- All configs inherit from `BaseConfig` for YAML/JSON serialization

**Key Design Decisions:**
- Tier 1 variables hardcoded as default: temperature, dew_point_temperature, sea_level_pressure, wind_speed, wind_direction, relative_humidity
- Default completeness threshold: 60% for all variables (adjustable)
- Default North America geographic bounds: lat (24.0, 50.0), lon (-130.0, -60.0)
- Default window size: 168 hours (1 week), stride: 24 hours
- Default split strategy: "simulated" (Strategy D from SPEC.md)
- Default masking strategy: "realistic" (based on observed gap patterns)
- Validation enforces: completeness 0-100%, ratios sum to 1.0, min < max for ranges/gaps

**Test Coverage:**
- Default values for all 5 config classes
- Custom values and overrides
- Validation errors (invalid enums, out-of-range values, ratio constraints)
- YAML roundtrip serialization
- Nested configuration serialization (DataConfig contains 4 sub-configs)

**Files Modified:**
- `src/weather_imputation/config/data.py` (created, 337 lines)
- `tests/test_config.py` (added 17 tests, now 31 tests total)
- `TODO.md` (marked TASK-002 as DONE)

**Confidence:** KNOWN - Directly implements SPEC.md section 3.4 requirements, validates per FR-003 through FR-007.

---

### TASK-003: Model Configuration Classes

**Status:** Completed

**Implementation:**
- Created `src/weather_imputation/config/model.py` with 6 configuration classes:
  - `ModelConfig`: Base configuration for all imputation models (model_type, n_variables)
  - `LinearInterpolationConfig`: Linear interpolation baseline (max_gap_size)
  - `SplineInterpolationConfig`: Akima spline interpolation (spline_order, max_gap_size)
  - `MICEConfig`: Multiple Imputation by Chained Equations (n_iterations, n_imputations, predictor_method)
  - `SAITSConfig`: Self-Attention Imputation for Time Series (n_layers, n_heads, d_model, d_ff, dropout, mit_weight, ort_weight, position encoding, wind encoding, metadata conditioning)
  - `CSDIConfig`: Conditional Score-based Diffusion Imputation (n_diffusion_steps, noise_schedule, beta_start/end, denoising network architecture, sampling strategy, n_samples)
- Added 14 comprehensive tests to `tests/test_config.py` (all passing, now 45 tests total)
- Updated `src/weather_imputation/config/__init__.py` to export new classes

**Key Design Decisions:**
- Classical methods (linear, spline, MICE) have minimal hyperparameters (no-train or simple sklearn wrappers)
- SAITS defaults match paper recommendations: 2 layers, 4 heads, d_model=128, d_ff=512
- CSDI defaults: 50 diffusion steps (cost vs quality tradeoff), cosine noise schedule, DDPM sampling
- Circular wind encoding enabled by default for SAITS (sin/cos components)
- Station metadata conditioning disabled by default (can be enabled for experiments)
- Cross-field validation using `@model_validator(mode="after")`:
  - SAITS/CSDI: d_model must be divisible by n_heads (required for multi-head attention)
  - CSDI: beta_end must be > beta_start (noise schedule monotonicity)

**Test Coverage:**
- Default values for all 6 model config classes
- Custom values and overrides
- Validation errors (invalid enums, out-of-range values, cross-field constraints)
- YAML roundtrip serialization
- Comprehensive coverage of SAITS and CSDI hyperparameters

**Files Modified:**
- `src/weather_imputation/config/model.py` (created, 325 lines)
- `src/weather_imputation/config/__init__.py` (added model config exports)
- `tests/test_config.py` (added 14 tests, now 45 tests total)
- `TODO.md` (marked TASK-003 as DONE)

**Lessons Learned:**
- Pydantic v2 `@model_validator(mode="after")` is required for cross-field validation where both fields are set simultaneously in __init__
- `@field_validator` with `info.data` doesn't work reliably for cross-field checks due to validation order
- Test case with d_model=100, n_heads=4 was incorrectly passing because 100 % 4 == 0 (divisible!)
  - Fixed to use d_model=101 (101 % 4 == 1, not divisible)

**Confidence:** KNOWN - Directly implements SPEC.md section 3.4 model configurations, matches paper specifications for SAITS/CSDI hyperparameters.

---

### TASK-004: Training Configuration Classes

**Status:** Completed

**Implementation:**
- Created `src/weather_imputation/config/training.py` with 5 configuration classes:
  - `OptimizerConfig`: Optimizer settings (type, learning_rate, weight_decay, betas, momentum, gradient clipping)
  - `SchedulerConfig`: Learning rate scheduler (cosine/plateau/step/onecycle, warmup, patience, decay factors)
  - `EarlyStoppingConfig`: Early stopping logic (enabled, patience, min_delta, monitor metric, mode)
  - `CheckpointConfig`: Model checkpointing (save frequency by epochs/minutes, keep_last_n, save_best)
  - `TrainingConfig`: Main configuration aggregating all sub-configs + batch_size, max_epochs, device, mixed_precision, compile_model
- Added 17 comprehensive tests to `tests/test_config.py` (all passing, now 62 tests total)
- Updated `src/weather_imputation/config/__init__.py` to export new classes

**Key Design Decisions:**
- Default optimizer: AdamW with lr=1e-3, weight_decay=1e-4 (proven effective for transformers)
- Default scheduler: Cosine annealing with 5-epoch warmup (smooth convergence)
- Default gradient clipping: norm=1.0 (prevents instability, especially for CSDI diffusion)
- Default checkpointing: every 1 epoch + every 30 minutes (NFR-009: spot preemption resilience)
- Keep last 3 checkpoints by default (balance storage vs safety)
- Mixed precision enabled by default (NFR-004: faster training on modern GPUs)
- torch.compile disabled by default (stability concerns, can enable for production)
- Validation enforces at least one checkpoint method enabled (prevent data loss)
- Cross-field validation for beta coefficients (must be in [0, 1))

**Implementation Notes:**
- Used modern Python 3.10+ type annotations: `float | None` instead of `Optional[float]`
- CheckpointConfig has `@model_validator(mode="after")` to ensure at least one checkpoint method is enabled
- OptimizerConfig has `@field_validator` for beta validation (both values must be in [0, 1))
- Supports 5 scheduler types: cosine (default), plateau, step, onecycle, none
- Supports 3 optimizer types: adamw (default), adam, sgd
- Monitor metrics for early stopping/checkpointing: val_loss, val_rmse, val_mae, val_r2

**Test Coverage:**
- Default values for all 5 config classes
- Custom values and overrides
- Validation errors (invalid ranges, cross-field constraints)
- YAML roundtrip serialization
- Nested configuration serialization (TrainingConfig contains 4 sub-configs)
- Edge cases (all checkpoint methods disabled, invalid beta coefficients)

**Files Modified:**
- `src/weather_imputation/config/training.py` (created, 219 lines)
- `src/weather_imputation/config/__init__.py` (added training config exports)
- `tests/test_config.py` (added 17 tests, now 62 tests total)
- `TODO.md` (marked TASK-004 as DONE)

**Lessons Learned:**
- Pydantic Field validators with `gt=0.0` (greater than) allow exactly 0.0, must use `ge=0.0` (greater or equal) or `gt=0.0` depending on intent
- For checkpoint frequency, better to validate at least one method enabled rather than setting defaults that could all be zero
- Modern type hints (`X | None`) are preferred over `Optional[X]` for Python 3.10+
- Ruff auto-fix handles UP045 (modernize type hints) cleanly

**Confidence:** KNOWN - Directly implements SPEC.md section 3.4 training configuration and NFR-004, NFR-006, NFR-009 requirements. Follows best practices from PyTorch training literature (gradient clipping, mixed precision, warmup).

---

### TASK-005: Evaluation Configuration Classes

**Status:** Completed

**Implementation:**
- Created `src/weather_imputation/config/evaluation.py` with 5 configuration classes:
  - `MetricConfig`: Point metrics (RMSE, MAE, Bias, R²) and probabilistic metrics (CRPS, calibration, coverage)
  - `StratificationConfig`: Stratified evaluation by gap length, season, variable, extremes
  - `StatisticalTestConfig`: Wilcoxon signed-rank test, Bonferroni correction, Cohen's d, bootstrap CI
  - `DownstreamValidationConfig`: Degree days (heating/cooling/growing) and extreme event detection
  - `EvaluationConfig`: Main configuration aggregating all sub-configs + output settings
- Added 14 comprehensive tests to `tests/test_config.py` (all passing, now 76 tests total)
- Updated `src/weather_imputation/config/__init__.py` to export new classes

**Key Design Decisions:**
- Point metrics enabled by default (RMSE, MAE, Bias, R²) - required for all methods
- Probabilistic metrics disabled by default (CRPS, calibration, coverage) - only for CSDI/ensemble methods
- All stratification dimensions enabled by default (gap length, season, variable, extremes)
- Default gap length bins: [1, 6, 24, 72, 168] hours (short, medium, long, very long gaps)
- Default extreme percentiles: 5th and 95th (standard practice)
- Default statistical tests: Wilcoxon + Bonferroni + Cohen's d + bootstrap CI (all enabled)
- Degree days enabled by default (FR-015: SHOULD), extreme events disabled (FR-016: COULD)
- Default degree day thresholds from meteorology standards (18°C heating/cooling, 10°C growing)
- Default output format: parquet (efficient for large result tables)
- Save predictions and stratified results by default (enables re-analysis without re-running)

**Implementation Notes:**
- Implements FR-011 through FR-016 from SPEC.md
- MetricConfig validates confidence levels are in (0, 1) exclusive
- StratificationConfig validates gap_length_bins are sorted and positive
- Default n_samples=100 for CSDI probabilistic metrics (balances cost vs uncertainty quality)
- Default confidence_levels=[0.50, 0.90, 0.95] for prediction intervals
- Default alpha=0.05 for statistical significance tests (standard p-value threshold)
- Default n_bootstrap_samples=1000 (sufficient for stable CI estimates)
- Default heat_wave_threshold=35°C, cold_snap_threshold=-10°C (reasonable for North America)
- Default event duration=3 days (meteorological definition)

**Test Coverage:**
- Default values for all 5 config classes
- Custom values and overrides
- Validation errors (invalid confidence levels, unsorted bins, out-of-range percentiles)
- YAML roundtrip serialization
- Nested configuration serialization (EvaluationConfig contains 4 sub-configs)
- Edge cases (confidence levels at boundaries, negative gap bins)

**Files Modified:**
- `src/weather_imputation/config/evaluation.py` (created, 256 lines)
- `src/weather_imputation/config/__init__.py` (added evaluation config exports)
- `tests/test_config.py` (added 14 tests, now 76 tests total)
- `TODO.md` (marked TASK-005 as DONE)

**Lessons Learned:**
- Pydantic field validators can access the entire list/value and validate element-wise
- For multi-line field descriptions, use parentheses to wrap the string for better readability
- Ruff E501 enforces 100 character line limit - use parentheses for long descriptions
- Configuration classes should provide sensible defaults that work for most use cases
- Probabilistic metrics should be opt-in (require more compute, only apply to certain models)
- Stratification dimensions should be opt-out (important for comprehensive evaluation)

**Confidence:** KNOWN - Directly implements SPEC.md section 3.2 (evaluation component), FR-011 through FR-016. Defaults based on meteorological standards and statistical best practices.

---

### TASK-006: GHCNh Variable Extraction Utilities

**Status:** Completed

**Implementation:**
- Added `extract_tier1_variables()` function to `src/weather_imputation/data/ghcnh_loader.py`:
  - Extracts Tier 1 weather variables (6 core variables: temperature, dew_point_temperature, sea_level_pressure, wind_speed, wind_direction, relative_humidity)
  - Extracts all 6 attributes for each variable: value, Quality_Code, Measurement_Code, Report_Type_Code, Source_Code, units
  - Includes primary columns: STATION, Station_name, DATE, LATITUDE, LONGITUDE
  - Handles missing columns gracefully (only includes columns that exist in DataFrame)
  - Supports extracting all Tier 1 variables (default) or a custom subset
- Created comprehensive test suite in `tests/test_ghcnh_loader.py` with 11 tests:
  - Default extraction (all Tier 1 variables)
  - Subset extraction (specific variables)
  - Single variable extraction
  - Missing columns handling
  - Empty DataFrame handling
  - No matching columns handling
  - Data value preservation
  - Row order preservation
  - Null value handling
  - Constants validation (TIER1_VARIABLES, VARIABLE_SUFFIXES)

**Key Design Decisions:**
- Function returns empty DataFrame if no columns match (rather than raising error)
- Only selects columns that exist in input DataFrame (defensive programming)
- Uses existing constants TIER1_VARIABLES and VARIABLE_SUFFIXES for consistency
- Logs warning when no requested columns are found
- Does not modify input DataFrame (pure function)

**Test Coverage:**
- All 11 tests passing
- Covers happy path, edge cases, error cases
- Validates data integrity (values, order, nulls preserved)
- Validates constants match SPEC.md requirements

**Files Modified:**
- `src/weather_imputation/data/ghcnh_loader.py` (added extract_tier1_variables function, ~50 lines)
- `tests/test_ghcnh_loader.py` (created, 259 lines with 11 tests)
- `TODO.md` (marked TASK-006 as DONE)

**Lessons Learned:**
- Polars `.select()` naturally handles column subset selection
- Building column list dynamically allows for missing columns without errors
- Ruff's SIM300 (Yoda condition) prefers `expected == actual` over `actual == expected`
- Pre-existing line length violations in file are not blockers for new code

**Confidence:** KNOWN - Implements FR-004 from SPEC.md (6-variable imputation), leverages well-documented GHCNh schema with 6 attributes per variable. Tests validate all requirements.
