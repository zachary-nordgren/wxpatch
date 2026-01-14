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
    Optimized to use Polars directly for reading/writing, eliminating redundant string operations.
    Metadata updates are accumulated in memory and written in batch later.
    """
    import polars as pl
    from io import StringIO
    from csv_merger import get_canonical_headers

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
                # Read existing file directly with Polars (no intermediate string)
                try:
                    with gzip.open(station_path, "rb") as f:
                        existing_df = pl.read_csv(
                            f,
                            has_header=True,
                            quote_char='"',
                            ignore_errors=True,
                            infer_schema_length=0,
                            try_parse_dates=False,
                        )
                except Exception as read_e:
                    # Fallback: read as string if direct read fails
                    logger.debug(f"Direct Polars read failed, using string fallback: {read_e}")
                    with gzip.open(
                        station_path, "rt", encoding="utf-8", errors="replace"
                    ) as f:
                        existing_content = f.read()
                    merged_content = merge_csv_data_with_polars(existing_content, content)
                    with gzip.open(station_path, "wt", encoding="utf-8") as f:
                        f.write(merged_content)
                    elapsed = time.time() - start_time
                    logger.debug(f"Merged and wrote {station_id} in {elapsed:.2f}s")
                    continue

                # Parse new content
                new_df = pl.read_csv(
                    StringIO(content),
                    has_header=True,
                    quote_char='"',
                    ignore_errors=True,
                    infer_schema_length=0,
                    try_parse_dates=False,
                )

                # Handle DATE column if present
                if "DATE" in existing_df.columns:
                    existing_df = existing_df.with_columns(
                        [pl.col("DATE").str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False)]
                    )
                if "DATE" in new_df.columns:
                    new_df = new_df.with_columns(
                        [pl.col("DATE").str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False)]
                    )

                # Fill nulls and cast to strings (except DATE)
                existing_cols = [col for col in existing_df.columns if col != "DATE"]
                new_cols = [col for col in new_df.columns if col != "DATE"]
                if existing_cols:
                    existing_df = existing_df.with_columns(
                        [pl.col(existing_cols).fill_null("").cast(pl.Utf8)]
                    )
                if new_cols:
                    new_df = new_df.with_columns(
                        [pl.col(new_cols).fill_null("").cast(pl.Utf8)]
                    )

                # Merge dataframes
                merged_df = (
                    pl.concat([existing_df, new_df], how="diagonal")
                    .unique()
                    .fill_null("")
                )

                # Get canonical headers and reorder
                canonical_headers = get_canonical_headers(merged_df.columns)
                merged_df = merged_df.select([pl.col(col) for col in canonical_headers])

                # Format DATE column back to string if present
                if "DATE" in merged_df.columns and merged_df["DATE"].dtype == pl.Datetime:
                    merged_df = merged_df.with_columns(
                        [pl.col("DATE").dt.strftime("%Y-%m-%dT%H:%M:%S")]
                    )

                # Cast all to strings for consistent output
                merged_df = merged_df.select(
                    [pl.col(col).cast(pl.Utf8) for col in merged_df.columns]
                )

                # Write directly to gzipped file
                with gzip.open(station_path, "wb") as f:
                    merged_df.write_csv(f, quote_style="necessary")

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
