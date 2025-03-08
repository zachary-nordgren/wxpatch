# NOAA Weather Data Processor

## Overview

This project downloads and processes NOAA Global Hourly weather data. It downloads compressed archives from NOAA's servers, extracts station data, and merges observations chronologically for analysis.

## Features

- Downloads and processes NOAA Global Hourly weather data archives
- Multi-threaded processing for improved performance
- Only downloads/processes new or updated files
- Tracks station metadata and observation counts
- Handles data in CSVs with all their annoying quirks
- Stores processed data in compressed format
- Has moderately decent error recovery

## Requirements

- Python 3.6+
- Required packages: `requests`, `tqdm`
- Optional packages for better performance:
  - `polars`: Makes CSV processing way faster (seriously, install this)
  - `psutil`: Checks if your computer can handle the load

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/noaa-weather-processor.git
cd noaa-weather-processor

# Set up virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install requests tqdm
pip install polars psutil  # Optional but recommended
```

## Usage

```bash
# Basic usage - download and process everything
python main.py

# Just download the archives without processing
python main.py --download-only

# Process archives already downloaded without getting new ones
python main.py --process-only

# Only process specific years (comma-separated, ranges with colon)
python main.py --year-filter 2020,2021,2022
python main.py --year-filter 2010:2015,2020  # 2010-2015 inclusive, plus 2020

# Process newest years first (default is oldest first)
python main.py --newest-first

# Sort everything chronologically when done
python main.py --sort-chronologically

# Update observation counts in metadata
python main.py --update-counts

# Try to fix corrupted metadata file
python main.py --recover-metadata

# Control parallelism (default is 6)
python main.py --max-workers 4

# Logging options
python main.py --verbose  # More logging details
python main.py --quiet    # Less logging details
```

## How It Works

1. **Downloading**: Gets NOAA archive files (yearly `.tar.gz` files)
2. **Extraction**: Pulls out station data files from the archives
3. **Processing**: Merges station data chronologically, separates metadata
4. **Storage**: Compresses merged station data and maintains metadata index

## Directory Structure

```
data/
  ├── raw/           # Downloaded .tar.gz archives from NOAA
  └── merged/        # Processed station data
      ├── wx_info.csv  # Station metadata and observation counts
      └── *.csv.gz     # Individual compressed station data files
```

## Notes

- Setting `--max-workers` too high will crash your computer
- You need about 30GB disk space per decade of data
- The first run takes forever, later updates are faster
