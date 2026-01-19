# Changelog

## [0.2.3] - 2026-01-18 - Schema Inconsistency Fixes

### Bug Fixes

- **Fixed Struct-type quality code columns causing concat errors**
  - Some parquet files have quality code columns as `Struct({'member0': Int32, 'member1': String})` instead of simple `Int32`
  - Schema varies across years for the same station (e.g., 2020 has Struct, 2021 has Int32)
  - Added `_normalize_schema()` function to extract integer values from Struct columns before concatenation
  - Fixes error: `must specify one field in the struct`
  - Affected stations include MXM00068550 and others with mixed year schemas

- **Fixed string-typed numeric columns causing mean() errors**
  - Some parquet files have numeric columns (temperature, dew_point, etc.) stored as String type
  - Added schema normalization to cast string columns to Float64 before processing
  - Fixes error: `'mean' operation not supported for dtype 'str'`
  - Affected stations include MXM00076856 (2009 data has string-typed temperature)

- **Fixed quality code type mismatch in is_in() operations**
  - Quality codes can be Int32 or String depending on file format
  - Added `_get_valid_quality_codes()` and `_get_invalid_quality_codes()` helpers
  - Automatically selects appropriate comparison values based on column dtype
  - Fixes error: `'is_in' cannot check for List(String) values in Int32 data`

### Internal Changes

- Added `_normalize_schema()` in `ghcnh_loader.py` - normalizes DataFrame schema before concatenation
- Added `_normalize_quality_column()` in `stats.py` - extracts Int32 from Struct quality columns
- Split `VALID_QUALITY_CODES` into `VALID_QUALITY_CODES_INT` and `VALID_QUALITY_CODES_STR`
- Split `INVALID_QUALITY_CODES` into `INVALID_QUALITY_CODES_INT` and `INVALID_QUALITY_CODES_STR`

---

## [0.2.2] - 2026-01-18 - Hourly Record Filtering and Incremental Updates

### New Features

- **Hourly record type filtering in metadata computation**
  - Metadata now filters to only hourly instrument type records by default
  - Excludes FM-12 SYNOP (3-hourly summaries) and other aggregate record types
  - Includes FM-15 METAR, FM-16 SPECI, SAO, AUTO, and other hourly sources
  - Added `HOURLY_REPORT_TYPES` and `SUMMARY_REPORT_TYPES` constants to `ghcnh_loader.py`
  - Added `filter_hourly_records()`, `count_report_types()`, `is_hourly_report_type()` functions

- **Report type counts in metadata**
  - Metadata now includes `report_type_counts` field with count of each report type per station
  - Includes `total_records_all_types` and `records_excluded_by_filter` for transparency
  - Report type counts are stored as JSON in parquet/CSV output

- **Incremental metadata updates**
  - New `--incremental` flag for `compute_metadata.py`
  - Only recomputes metadata for stations whose parquet files have been modified
  - Compares file modification time against `metadata_computed_at` timestamp
  - Merges updated stations with existing unchanged metadata
  - Significantly faster when only a few stations have new data
  - Usage: `uv run python src/scripts/compute_metadata.py compute --incremental`

- **New functions in `metadata.py`**
  - `get_station_file_mtime()` - Get latest file modification time for a station
  - `get_stations_needing_update()` - Identify stations requiring metadata recomputation
  - `compute_all_metadata_incremental()` - Compute metadata with incremental updates

---

## [0.2.1] - 2026-01-18 - Metadata Fixes and Station Inventory Integration

### Bug Fixes

- **Fixed DATE column parsing in metadata computation**
  - GHCNh parquet files store DATE as string (ISO format like "2024-01-01T12:00:00")
  - Now properly parses string dates to datetime before using `.dt.year()` operations
  - Fixes error: `year operation not supported for dtype str`

- **Fixed column name mismatches**
  - GHCNh parquet files use mixed-case column names (e.g., `Latitude` not `LATITUDE`)
  - Now handles both uppercase and mixed-case variants

- **Fixed file naming convention in loader**
  - Parquet files are named `GHCNh_{station_id}_{year}.parquet`
  - `list_all_stations()` now correctly extracts station IDs from filenames
  - `list_available_years()` now finds files with the correct naming pattern

- **Fixed station list CSV parsing**
  - Added schema overrides for WMO_ID, ICAO, and other mixed-type columns
  - Handles Windows line ending artifacts in column names

- **Fixed schema mismatch when concatenating multi-year station data**
  - GHCNh parquet files changed schema in 2025 (added Year, Month, Day, Hour, Minute columns)
  - Changed `pl.concat` from `vertical_relaxed` to `diagonal_relaxed` to handle different column sets
  - Added unified date handling: combines DATE column (older files) with Year/Month/Day/Hour/Minute (newer files)

### New Features

- **NOAA Station Inventory Integration**
  - `compute_metadata.py` now automatically enriches metadata from NOAA's official station list
  - Downloads `ghcnh-station-list.csv` from NOAA if not present locally
  - Fills in missing latitude, longitude, elevation from inventory
  - Adds new fields: `state`, `wmo_id`, `icao_code`
  - Example: stations with null lat/lon in parquet files now get coordinates from inventory

- **New function: `download_station_list()`**
  - Downloads official GHCNh station list from NOAA
  - Saves to `data/raw/ghcnh/ghcnh-station-list.csv`

- **New function: `enrich_metadata_from_station_list()`**
  - Joins computed metadata with NOAA station inventory
  - Fills missing values for lat/lon/elevation/station_name
  - Adds WMO ID and ICAO code fields

---

## [0.2.0] - 2026-01-18 - Architecture Migration

### Breaking Changes

#### Complete Architecture Migration to GHCNh Parquet

This release completely restructures the project from legacy ISD tar.gz-based processing to a modern GHCNh parquet-based architecture for weather imputation research.

**Removed (~2,600 lines of legacy code):**
- `downloader.py` - ISD tar.gz downloader
- `processor_core.py` - Tar.gz extraction queue
- `csv_merger.py` - ISD CSV filtering/merging
- `update_dataset.py` - Incremental ISD updates
- `verification.py` - CSV timestamp verification
- `combine_datasets.py` - Merge two datasets
- `main.py` - ISD orchestration
- `repair_stations.py` - ISD datetime fix utility
- `metadata_manager.py` - SQLite-based metadata
- `file_io.py` - Legacy file utilities
- `utils.py` - Legacy utilities
- `config.py` - Legacy configuration

**New Package Structure:**
```
src/
├── weather_imputation/           # Main package
│   ├── config/                   # paths.py, download.py
│   ├── data/                     # ghcnh_loader.py, metadata.py, stats.py, cleaning.py
│   ├── models/                   # base.py, classical/{linear.py, spline.py, mice.py}
│   ├── training/                 # trainer.py, callbacks.py, checkpoint.py
│   ├── evaluation/               # metrics.py, statistical.py, stratified.py
│   └── utils/                    # progress.py, parsing.py, filesystem.py, system.py
└── scripts/                      # ghcnh_downloader.py, compute_metadata.py, clean_metadata.py
```

**Key Changes:**
- Parquet files kept separate by station/year (no merging)
- SQLite metadata → Parquet metadata with cleaning/deduplication
- tqdm → rich.progress for all progress display
- `uv` for dependency locking

### New Features

#### New CLI Scripts

- **`compute_metadata.py`** - Compute station metadata from parquet files
  ```bash
  uv run python src/scripts/compute_metadata.py compute
  uv run python src/scripts/compute_metadata.py compute --years 2023,2024
  uv run python src/scripts/compute_metadata.py show
  uv run python src/scripts/compute_metadata.py stats
  ```

- **`clean_metadata.py`** - Clean and deduplicate station metadata
  ```bash
  uv run python src/scripts/clean_metadata.py clean
  uv run python src/scripts/clean_metadata.py clean --dry-run
  uv run python src/scripts/clean_metadata.py duplicates
  uv run python src/scripts/clean_metadata.py validate
  uv run python src/scripts/clean_metadata.py report
  ```

#### Metadata Cleaning Features

- **Duplicate detection** - Find stations with same location (lat/lon within threshold)
- **Duplicate merging** - Merge duplicate stations with configurable strategy
- **Coordinate lookup** - Fill missing lat/lon/elevation from station list
- **Coordinate validation** - Validate coordinate ranges, flag outliers
- **Name normalization** - Clean and standardize station names

#### Rich Progress Display

- All progress bars now use Rich library instead of tqdm
- Download progress with speed and ETA
- Processing progress with completed/total counts
- Spinners for indeterminate operations
- Formatted summary tables

#### Imputation Framework (Scaffolding)

- **BaseImputer** protocol for consistent imputation interface
- **Classical imputers**: LinearInterpolationImputer, AkimaSplineImputer, MICEImputer
- **Training infrastructure**: Trainer, callbacks, checkpoint management
- **Evaluation framework**: RMSE, MAE, R², CRPS metrics, stratified analysis

---

## [0.1.x] - Previous Releases

### New Features

#### 2026-01-18 - New GHCNh Parquet Downloader

- **Added new downloader for GHCNh (Global Historical Climatology Network hourly) data**
  - The old data source (ISD Global Hourly CSV archives) has reached end of life
  - New source: NOAA GHCNh parquet files via S3-style API
  - Downloads individual station parquet files directly (no tar.gz extraction needed)
  - Data spans 1790 to present with 237 years available

- **New script: `ghcnh_downloader.py`**
  - Downloads parquet files to `data/raw/ghcnh/{year}/` directory structure
  - Filters to US stations by default (saves bandwidth)
  - Supports year ranges: `--year 2020:2024` or `--year 2020,2022,2024`

- **Enhanced download features:**
  - **Automatic retries** with exponential backoff
  - **Resume/restart capability** - tracks progress in `.download_state.json`
  - **Rich progress bars** with bandwidth estimates and ETA
  - **Concurrent downloads** - configurable parallel downloads (default: 12)
  - **State persistence** - interrupted downloads resume from where they left off

#### 2026-01-18 - GHCNh Downloader North America Filter Update

- **Expanded North America station filter to include US territories**
  - Added Puerto Rico (RQ) and US Virgin Islands (VQ) to the filter
  - Renamed constant from `US_STATION_PREFIXES` to `NORTH_AMERICA_PREFIXES`
  - Now includes: US, CA, MX, RQ, VQ

- **Added `--inventory` command to display station counts from NOAA**
  - Downloads official `ghcnh-station-list.csv` from NOAA
  - Shows total station counts worldwide and by country code
  - Displays North America breakdown (US, CA, MX, RQ, VQ)
  - Lists top 10 other countries for reference
  - Usage: `python ghcnh_downloader.py --inventory`

#### 2026-01-18 - GHCNh Downloader Audit and Clean Features

- **Fixed `--audit --fix` to report actual download results**
  - Previously showed "Fixed files have been downloaded" regardless of success/failure
  - Now tracks and reports: attempted, successful, and failed counts
  - Shows retry message if any downloads failed

- **Added `--clean` feature to remove unwanted files**
  - Identifies corrupted files (size mismatch with server)
  - Identifies files not matching station filter (e.g., non-North America stations)
  - Dry-run by default: shows what would be deleted without deleting
  - Use `--clean --delete` to actually delete files
  - Reports total files and bytes to be deleted/deleted

- **Usage:**
  ```bash
  # Check for corrupted/non-matching files (dry run)
  python ghcnh_downloader.py --clean

  # Actually delete corrupted/non-matching files
  python ghcnh_downloader.py --clean --delete

  # Clean specific years only
  python ghcnh_downloader.py --clean --year 2020:2024
  ```

#### 2026-01-18 - GHCNh Downloader Improvements

- **Added automatic retry queue for failed downloads**
  - Failed files are now automatically retried in a second pass after the main download
  - Brief pause between passes to give server time to recover
  - Clear reporting of files that still fail after retry

- **Improved progress bar display**
  - Narrower progress bar (100 columns) to prevent log line wrapping issues
  - Shows current/max concurrent downloads (e.g., `4/12`)
  - Cleaner format: `Year 2019 | 78%|████| 4357/5610 [5.36files/s] 4/12 | 1.7 MB/s | F:0`

- **Added file logging**
  - All logs now written to `src/ghcnh_downloader.log` in addition to console
  - File logging always captures INFO level and above
  - Useful for debugging download issues after the fact

- **Simplified concurrency control**
  - Replaced adaptive rate limiter with simpler semaphore-based concurrency controller
  - Fixed concurrency tracking to accurately show active downloads
  - Removed unused rate limit recovery/backoff logic

- **Usage:**
  ```bash
  # Download all years (US stations only)
  python ghcnh_downloader.py

  # Download specific year
  python ghcnh_downloader.py --year 2024

  # Download year range
  python ghcnh_downloader.py --year 2020:2024

  # Download all stations globally
  python ghcnh_downloader.py --all-stations

  # List available years
  python ghcnh_downloader.py --list-years

  # Check download status
  python ghcnh_downloader.py --status

  # Force re-download
  python ghcnh_downloader.py --force
  ```

- **Note:** This is a new downloader for raw data. Processing/merging scripts will need updates to handle the new GHCNh parquet format (234 columns, pipe-separated quality codes vs. the old ISD format).

