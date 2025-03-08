#!/usr/bin/env python3
"""
Functions for verifying data completeness through timestamp checking.
"""
import os
import logging
import gzip
import csv
from io import StringIO
from typing import Set

from config import MERGED_DIR

logger = logging.getLogger("weather_processor")


def extract_sample_timestamps(content: str) -> Set[str]:
    """
    Extract just first and last timestamps from CSV content.
    Handles quoted fields properly.
    """

    # Quick check if content is empty
    if not content:
        return set()

    # Handle if content ends with newline
    if content.endswith("\n"):
        content = content.rstrip("\n")

    # Get the first line (header) and first data line
    lines = content.splitlines()
    if len(lines) < 2:  # Need at least header and one data line
        return set()

    # Parse the header with csv module to handle quotes properly
    header_reader = csv.reader(StringIO(lines[0]), quotechar='"')
    header = next(header_reader)

    # Find date column index
    date_col_idx = -1
    for i, field in enumerate(header):
        if field in ["DATE", "DATETIME", "DATE_TIME", "TIME"]:
            date_col_idx = i
            break

    if date_col_idx == -1:
        # Extra debug info to help understand what's in the header
        logger.warning(f"No timestamp column found in CSV. Header: {header[:10]}...")
        return set()

    # Extract timestamps using the csv module
    timestamps = set()

    # First data line
    if len(lines) >= 2:
        first_line_reader = csv.reader(StringIO(lines[1]), quotechar='"')
        try:
            first_row = next(first_line_reader)
            if date_col_idx < len(first_row):
                timestamps.add(first_row[date_col_idx])
        except Exception as e:
            logger.debug(f"Error parsing first data line: {e}")

    # Last data line (if different from first)
    if len(lines) > 2:
        last_line_reader = csv.reader(StringIO(lines[-1]), quotechar='"')
        try:
            last_row = next(last_line_reader)
            if date_col_idx < len(last_row):
                timestamps.add(last_row[date_col_idx])
        except Exception as e:
            logger.debug(f"Error parsing last data line: {e}")

    return timestamps


def verify_timestamps_in_merged_file(station_id: str, timestamps: Set[str]) -> bool:
    """
    Check if all sample timestamps exist in the merged file.
    Returns True if all timestamps are found.
    """
    if not timestamps:
        return False

    merged_path = MERGED_DIR / f"{station_id}.csv.gz"
    if not merged_path.exists():
        return False

    try:
        with gzip.open(merged_path, "rt", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Check each timestamp
        for timestamp in timestamps:
            if timestamp not in content:
                logger.debug(
                    f"Timestamp {timestamp} not found in merged file for {station_id}"
                )
                return False

        logger.debug(
            f"All {len(timestamps)} timestamps found in merged file for {station_id}"
        )
        return True

    except Exception as e:
        logger.error(f"Error verifying timestamps for {station_id}: {e}")
        return False


def should_process_station_file(
    station_id: str, content: str, last_modified: float
) -> bool:
    """
    Determine if a station file needs processing by checking timestamps.
    """
    merged_path = MERGED_DIR / f"{station_id}.csv.gz"

    # Quick check based on file existence and modification date
    if not merged_path.exists() or last_modified > os.path.getmtime(merged_path):
        return True

    # Extract sample timestamps from content
    sample_timestamps = extract_sample_timestamps(content)
    if not sample_timestamps:
        logger.warning(
            f"Could not extract timestamps from {station_id}, defaulting to processing"
        )
        return True

    # Check if all timestamps exist in the merged file
    if not verify_timestamps_in_merged_file(station_id, sample_timestamps):
        logger.debug(
            f"Some timestamps from {station_id} not found in merged file, needs processing"
        )
        return True

    logger.debug(f"All timestamps from {station_id} found in merged file, skipping")
    return False
