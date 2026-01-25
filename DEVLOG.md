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
