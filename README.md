# NOAA Weather Data Processor

## Overview

This project downloads and processes NOAA Global Hourly weather data to create a complete, accurate historical weather dataset for the continental USA. It downloads compressed archives from NOAA's servers, extracts station data, merges observations chronologically, and maintains comprehensive metadata including data quality metrics and statistics.

## Features

- **Efficient Processing**: Optimized with Polars for fast CSV/Parquet operations
- **Multi-threaded Processing**: Parallel processing for improved performance
- **Smart Updates**: Only downloads/processes new or updated files
- **Comprehensive Metadata**: Tracks station metadata, observation counts, data completeness, statistics, and quality metrics
- **Multiple Formats**: Supports both CSV.gz and Parquet formats
- **Data Quality Metrics**: Computes completeness percentages, statistics, gap analysis, and more
- **Automated Updates**: Simple command to update dataset from end of current data to present
- **Dataset Merging**: Tools to combine multiple merged datasets
- **Robust Error Handling**: Recovery mechanisms and validation

## Requirements

- Python 3.8+
- Required packages: `requests`, `tqdm`, `polars`
- Optional packages:
  - `psutil`: System resource checking

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/noaa-weather-processor.git
cd noaa-weather-processor

# Set up virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install requests tqdm polars
pip install psutil  # Optional but recommended
```

## Usage

### Basic Workflows

#### 1. Initial Setup - Process All Data

```bash
# Download and process all available data
python src/main.py

# Process newest years first (faster to see recent data)
python src/main.py --newest-first

# With more workers for faster processing
python src/main.py --max-workers 20
```

#### 2. Update Existing Dataset

```bash
# Automatically update from end of current data to present
python src/update_dataset.py

# Update without computing statistics (faster)
python src/update_dataset.py --no-stats

# Update with custom merged directory
python src/update_dataset.py --merged-dir data/merged-combined
```

#### 3. Combine Multiple Datasets

```bash
# Combine two merged datasets (e.g., merged/ and merged-1/)
python src/combine_datasets.py

# Combine and output as Parquet format (better performance)
python src/combine_datasets.py --parquet

# Custom directories
python src/combine_datasets.py --merged-dir data/merged --merged1-dir data/merged-1 --output-dir data/merged-combined
```

### Main Script Options

```bash
# Basic usage - download and process everything
python src/main.py

# Just download the archives without processing
python src/main.py --download-only

# Process archives already downloaded without getting new ones
python src/main.py --process-only

# Only process specific years (comma-separated, ranges with colon)
python src/main.py --year-filter 2020,2021,2022
python src/main.py --year-filter 2010:2015,2020  # 2010-2015 inclusive, plus 2020

# Process newest years first (default is oldest first)
python src/main.py --newest-first

# Sort everything chronologically when done
python src/main.py --sort-chronologically

# Update observation counts in metadata
python src/main.py --update-counts

# Try to fix corrupted metadata file
python src/main.py --recover-metadata

# Control parallelism (default is 14)
python src/main.py --max-workers 20

# Logging options
python src/main.py --verbose  # More logging details
python src/main.py --quiet    # Less logging details
```

## How It Works

1. **Downloading**: Gets NOAA archive files (yearly `.tar.gz` files) from the FTP server
2. **Extraction**: Pulls out station data files from the archives using producer/consumer pattern
3. **Processing**: 
   - Merges station data chronologically using Polars
   - Separates metadata from observation data
   - Deduplicates records
   - Optimized to read/write directly without intermediate string conversions
4. **Storage**: 
   - Compresses merged station data (CSV.gz or Parquet)
   - Maintains SQLite metadata database with comprehensive statistics
   - Exports metadata to CSV for compatibility

## Directory Structure

```
data/
  ├── raw/                    # Downloaded .tar.gz archives from NOAA
  ├── merged/                  # Processed station data
  │   ├── wx_metadata.db      # SQLite database with comprehensive metadata
  │   ├── wx_info.csv         # Station metadata and observation counts (CSV export)
  │   └── *.csv.gz            # Individual compressed station data files
  ├── merged-1/               # Alternative merged dataset (if applicable)
  ├── merged-combined/        # Combined datasets (from combine_datasets.py)
  └── temp/                   # Temporary files during processing
```

## Metadata

The system maintains comprehensive metadata for each station:

### Basic Metadata
- Station ID, name, location (latitude, longitude, elevation)
- Call sign

### Temporal Coverage
- First and last observation dates
- Total observation count
- Year-by-year observation counts

### Data Quality Metrics
- **Completeness**: Percentage of observations with data for each variable (TMP, DEW, SLP, WND, VIS, CIG)
- **Statistics**: Mean, std dev, min, max for temperature, dewpoint, and pressure
- **Gap Analysis**: Number of significant gaps (>24 hours) and maximum gap duration
- **Observation Frequency**: Average hours between observations
- **Timezone**: Approximate timezone based on longitude

This metadata is essential for:
- **Model Training**: Understanding data quality for imputation models
- **Data Validation**: Identifying stations with gaps or low quality
- **Feature Engineering**: Using station-specific statistics for normalization
- **Quality Assessment**: Understanding which variables are reliable for each station

## Performance Optimizations

The codebase has been optimized for performance:

- **Direct Polars I/O**: Reads/writes directly from gzip files without intermediate strings
- **Lazy Evaluation**: Uses Polars efficiently to minimize memory usage
- **Reduced I/O**: Eliminates redundant file reads
- **Parquet Support**: Optional Parquet format for 10-100x faster reads/writes

Expected performance improvements:
- **2-5x faster** processing with current optimizations
- **10-100x faster** with Parquet format (when implemented)

## Common Workflows

### Workflow 1: Initial Data Processing

```bash
# 1. Download and process all data (this takes a while!)
python src/main.py --newest-first

# 2. Sort all files chronologically
python src/main.py --sort-chronologically

# 3. Update observation counts
python src/main.py --update-counts
```

### Workflow 2: Combining Two Datasets

```bash
# Combine merged/ and merged-1/ into merged-combined/
# This also computes statistics for all stations
python src/combine_datasets.py --parquet
```

### Workflow 3: Regular Updates

```bash
# Update from end of current data to present
# Automatically determines what's needed
python src/update_dataset.py
```

### Workflow 4: Processing Specific Years

```bash
# Download and process only 2024 and 2025
python src/main.py --year-filter 2024,2025

# Or just download
python src/main.py --year-filter 2024,2025 --download-only

# Then process later
python src/main.py --year-filter 2024,2025 --process-only
```

## Notes

- **Performance**: Setting `--max-workers` too high will use excessive memory. Default (14) is usually optimal.
- **Disk Space**: You need about 30-50GB disk space per decade of data (raw + processed)
- **First Run**: The first run takes a long time (hours to days depending on data volume)
- **Updates**: Later updates are much faster, only processing new data
- **Parquet**: Consider using Parquet format for better performance, especially for model training
- **Statistics**: Computing statistics adds time but provides valuable metadata for analysis

## Troubleshooting

### Metadata Database Issues

```bash
# Try to recover corrupted metadata
python src/main.py --recover-metadata
```

### Processing Errors

- Check the log file: `src/weather_data_processing.log`
- Reduce `--max-workers` if running out of memory
- Use `--verbose` for more detailed error messages

### Update Issues

- If `update_dataset.py` can't find latest date, it will check station files (slower but works)
- Use `--no-stats` to skip statistics computation if it's taking too long
