#!/usr/bin/env python3
"""
Script to update the weather dataset from the end of current data to present.
Automatically determines what data is needed and downloads/processes only new data.

Usage:
    python src/update_dataset.py [--merged-dir <path>] [--compute-stats]
"""
import argparse
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm

from config import DATA_DIR, MERGED_DIR, RAW_DIR
from downloader import get_remote_file_list, download_archives
from file_io import setup_directories
from processor_core import merge_station_data
from metadata_manager import (
    get_db_connection,
    update_station_statistics,
    export_to_csv,
    initialize_database,
)

logger = logging.getLogger("weather_processor")


def get_latest_data_date(merged_dir: Path) -> Optional[datetime]:
    """
    Get the latest observation date from the merged dataset.
    
    Args:
        merged_dir: Path to merged directory
        
    Returns:
        Latest observation date or None if no data found
    """
    db_path = merged_dir / "wx_metadata.db"
    
    if not db_path.exists():
        logger.warning("Metadata database not found, checking station files...")
        return get_latest_date_from_files(merged_dir)
    
    try:
        initialize_database()
        with get_db_connection() as conn:
            cursor = conn.execute("""
                SELECT MAX(last_observation_date) 
                FROM stations 
                WHERE last_observation_date IS NOT NULL 
                  AND last_observation_date != ''
            """)
            result = cursor.fetchone()
            
            if result and result[0]:
                try:
                    # Parse the date string
                    latest_date = datetime.strptime(result[0], "%Y-%m-%dT%H:%M:%S")
                    logger.info("Latest data date from metadata: %s", latest_date)
                    return latest_date
                except ValueError:
                    logger.warning("Could not parse date from metadata: %s", result[0])
                    return get_latest_date_from_files(merged_dir)
            else:
                logger.info("No dates found in metadata, checking station files...")
                return get_latest_date_from_files(merged_dir)
    
    except Exception as e:
        logger.error("Error reading latest date from metadata: %s", e)
        return get_latest_date_from_files(merged_dir)


def get_latest_date_from_files(merged_dir: Path) -> Optional[datetime]:
    """
    Get the latest observation date by checking station files directly.
    This is a fallback if metadata doesn't have the date.
    
    Args:
        merged_dir: Path to merged directory
        
    Returns:
        Latest observation date or None if no data found
    """
    import gzip
    import polars as pl
    from io import StringIO
    
    logger.info("Scanning station files for latest date...")
    station_files = list(merged_dir.glob("*.csv.gz")) + list(merged_dir.glob("*.parquet"))
    
    if not station_files:
        logger.warning("No station files found")
        return None
    
    latest_date = None
    checked = 0
    
    # Sample a subset of files to find latest date (faster than checking all)
    sample_size = min(100, len(station_files))
    import random
    sample_files = random.sample(station_files, sample_size) if len(station_files) > sample_size else station_files
    
    for station_file in tqdm(sample_files, desc="Checking files", leave=False):
        try:
            if station_file.suffix == ".gz" or station_file.suffixes == [".csv", ".gz"]:
                with gzip.open(station_file, "rb") as f:
                    df = pl.read_csv(
                        f,
                        has_header=True,
                        quote_char='"',
                        ignore_errors=True,
                        infer_schema_length=0,
                        try_parse_dates=False,
                    )
            elif station_file.suffix == ".parquet":
                df = pl.read_parquet(station_file)
            else:
                continue
            
            if "DATE" in df.columns and df.shape[0] > 0:
                # Parse dates
                df = df.with_columns(
                    [pl.col("DATE").str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False)]
                )
                date_col = df["DATE"]
                valid_dates = date_col.filter(date_col.is_not_null())
                if len(valid_dates) > 0:
                    file_latest = valid_dates.max()
                    if file_latest is not None:
                        if latest_date is None or file_latest > latest_date:
                            latest_date = file_latest
                checked += 1
                
                # If we've checked enough files, break
                if checked >= sample_size:
                    break
        
        except Exception as e:
            logger.debug("Error reading %s: %s", station_file.name, e)
            continue
    
    if latest_date:
        logger.info("Latest data date from files: %s", latest_date)
    else:
        logger.warning("Could not determine latest date from files")
    
    return latest_date


def get_years_to_update(latest_date: Optional[datetime]) -> list:
    """
    Determine which years need to be downloaded based on latest data date.
    
    Args:
        latest_date: Latest observation date in dataset
        
    Returns:
        List of years (as strings) that need updating
    """
    current_date = datetime.now()
    current_year = current_date.year
    
    if latest_date is None:
        # No data exists, need to download everything
        logger.info("No existing data found, will download all available years")
        return []
    
    # Get the year of the latest data
    latest_year = latest_date.year
    
    # We need to update from latest_year onwards (in case there's partial data)
    # Also check current year and previous year (data might be delayed)
    years_to_update = []
    
    # Always check the year of latest data (might have new data)
    if latest_year <= current_year:
        years_to_update.append(str(latest_year))
    
    # Check subsequent years
    for year in range(latest_year + 1, current_year + 1):
        years_to_update.append(str(year))
    
    logger.info("Years to update: %s", ", ".join(years_to_update) if years_to_update else "All years")
    return years_to_update


def update_dataset(
    merged_dir: Path = MERGED_DIR,
    compute_stats: bool = True,
    max_workers: int = 14
) -> bool:
    """
    Update the dataset from the end of current data to present.
    
    Args:
        merged_dir: Path to merged directory
        compute_stats: Whether to compute statistics for updated stations
        max_workers: Maximum number of workers for processing
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Starting dataset update...")
    
    # Setup directories
    setup_directories()
    
    # Get latest data date
    logger.info("Determining latest data date...")
    latest_date = get_latest_data_date(merged_dir)
    
    if latest_date:
        logger.info("Current dataset ends at: %s", latest_date)
    else:
        logger.info("No existing data found, will download all available data")
    
    # Get list of available files from NOAA
    logger.info("Fetching available files from NOAA...")
    remote_files = get_remote_file_list()
    
    if not remote_files:
        logger.error("No files found on remote server")
        return False
    
    # Determine which years need updating
    years_to_update = get_years_to_update(latest_date)
    
    # Filter remote files to only those we need
    if years_to_update:
        # Filter to specific years
        files_to_download = [
            (filename, date, size) for filename, date, size in remote_files
            if filename.replace(".tar.gz", "") in years_to_update
        ]
        logger.info("Found %d files to check for years: %s", 
                   len(files_to_download), ", ".join(years_to_update))
    else:
        # Download all files (no existing data)
        files_to_download = remote_files
        logger.info("Checking all %d available files", len(files_to_download))
    
    # Download new/updated files
    # download_archives will only download files that are new or updated
    if files_to_download:
        logger.info("Downloading/updating archives...")
        downloaded = download_archives(files_to_download)
        logger.info("Downloaded/updated %d files", len(downloaded))
    else:
        logger.info("No new files to download")
        downloaded = []
    
    # Process downloaded files
    if downloaded:
        logger.info("Processing downloaded archives...")
        try:
            # Convert to Path objects if needed
            archive_paths = []
            for f in downloaded:
                if isinstance(f, (str, Path)):
                    archive_paths.append(Path(f))
                else:
                    archive_paths.append(f)
            
            # Process archives
            updated_stations = merge_station_data(
                archive_paths, max_workers=max_workers
            )
            
            logger.info("Updated %d stations", len(updated_stations))
            
            # Compute statistics for updated stations if requested
            if compute_stats:
                logger.info("Computing statistics for updated stations...")
                compute_statistics_for_stations(updated_stations, merged_dir)
            
            # Export metadata to CSV
            logger.info("Exporting metadata to CSV...")
            export_to_csv()
            
            logger.info("Dataset update completed successfully")
            return True
            
        except Exception as e:
            logger.error("Error processing archives: %s", e, exc_info=True)
            return False
    else:
        logger.info("No new data to process")
        return True


def compute_statistics_for_stations(station_ids: set, merged_dir: Path):
    """
    Compute statistics for a set of stations.
    
    Args:
        station_ids: Set of station IDs to update
        merged_dir: Path to merged directory
    """
    if not station_ids:
        return
    
    logger.info("Computing statistics for %d stations...", len(station_ids))
    
    success_count = 0
    failed_count = 0
    
    for station_id in tqdm(station_ids, desc="Computing statistics", unit="station"):
        try:
            # Try CSV.gz first, then Parquet
            csv_file = merged_dir / f"{station_id}.csv.gz"
            parquet_file = merged_dir / f"{station_id}.parquet"
            
            station_file = None
            if parquet_file.exists():
                station_file = parquet_file
            elif csv_file.exists():
                station_file = csv_file
            
            if station_file:
                update_station_statistics(station_id, station_file)
                success_count += 1
            else:
                logger.warning("Station file not found for %s", station_id)
                failed_count += 1
                
        except Exception as e:
            logger.error("Error computing statistics for %s: %s", station_id, e)
            failed_count += 1
    
    logger.info("Computed statistics for %d stations, %d failed", success_count, failed_count)


def main():
    """Main entry point for the update script."""
    parser = argparse.ArgumentParser(
        description="Update weather dataset from end of current data to present"
    )
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=MERGED_DIR,
        help="Path to merged directory (default: data/merged)",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip computing statistics (faster, but less metadata)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=14,
        help="Maximum number of workers for processing (default: 14)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Run update
    success = update_dataset(
        merged_dir=args.merged_dir,
        compute_stats=not args.no_stats,
        max_workers=args.max_workers
    )
    
    if success:
        logger.info("Update completed successfully")
    else:
        logger.error("Update completed with errors")
        exit(1)


if __name__ == "__main__":
    main()
