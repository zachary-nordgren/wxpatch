#!/usr/bin/env python3
"""
Functions for file I/O operations in the weather data processor.
"""
import gzip
import logging
import shutil
import time
from typing import Dict, Tuple
from tqdm import tqdm

from config import DATA_DIR, RAW_DIR, MERGED_DIR
from csv_merger import merge_csv_data_with_polars

logger = logging.getLogger("weather_processor")


def setup_directories():
    """Create necessary directories if they don't exist."""
    for directory in [DATA_DIR, RAW_DIR, MERGED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directories set up: {DATA_DIR}, {RAW_DIR}, {MERGED_DIR}")


def get_station_modified_dates() -> Dict[str, float]:
    """
    Get the last modified timestamp for each station's merged data file.
    Returns a dictionary mapping station IDs to their last modified timestamp.
    """
    last_modified = {}

    for file_path in MERGED_DIR.glob("*.csv.gz"):
        station_id = file_path.stem.replace(".csv", "")
        last_modified[station_id] = file_path.stat().st_mtime

    return last_modified


def write_station_data_to_disk(
    station_data: Dict[str, Tuple[str, Dict[str, str], bool]],
):
    """
    Write station data to disk and update metadata.
    Metadata updates are accumulated in memory and written in batch later.
    """

    for station_id, (content, metadata, is_new) in tqdm(
        station_data.items(), desc="Writing station files", unit="station", leave=False
    ):
        if not content:  # Skip if content was cleared to save memory
            continue

        start_time = time.time()
        station_path = MERGED_DIR / f"{station_id}.csv.gz"

        # Append or create station data file
        if station_path.exists():
            try:
                # Read existing content
                with gzip.open(
                    station_path, "rt", encoding="utf-8", errors="replace"
                ) as f:
                    existing_content = f.read()

                merged_content = merge_csv_data_with_polars(existing_content, content)

                # Write back merged content
                with gzip.open(station_path, "wt", encoding="utf-8") as f:
                    f.write(merged_content)

                elapsed = time.time() - start_time
                logger.debug(f"Merged and wrote {station_id} in {elapsed:.2f}s")

            except Exception as e:
                logger.error(
                    f"Error updating station file {station_id}: {e}", exc_info=True
                )
                # Create backup of problematic file
                if station_path.exists():
                    backup_path = station_path.with_suffix(".csv.gz.bak")
                    try:
                        shutil.copy(station_path, backup_path)
                        logger.info(
                            f"Created backup of problematic file at {backup_path}"
                        )
                    except Exception as backup_e:
                        logger.error(f"Failed to create backup: {backup_e}")

                # Try writing new content directly
                try:
                    with gzip.open(station_path, "wt", encoding="utf-8") as f:
                        f.write(content)
                    logger.info(
                        f"Wrote new content to {station_id} after merge failure"
                    )
                except Exception as write_e:
                    logger.error(f"Failed to write new content: {write_e}")
        else:
            # For new files, write the entire content
            try:
                with gzip.open(station_path, "wt", encoding="utf-8") as f:
                    f.write(content)

                elapsed = time.time() - start_time
                logger.debug(f"Created new file for {station_id} in {elapsed:.2f}s")

            except Exception as e:
                logger.error(f"Error creating new station file {station_id}: {e}")
