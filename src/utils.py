#!/usr/bin/env python3
"""
Utility functions for the NOAA weather data processor.
"""
import logging
import platform
import csv
import shutil
from pathlib import Path
from io import StringIO
from typing import List, Tuple, Optional
import psutil

logger = logging.getLogger("weather_processor")


def parse_year_filter(year_filter_str: str) -> List[str]:
    """
    Parse a year filter string that can contain individual years and ranges.

    Example formats:
    - Individual years: "2010,2012,2015"
    - Year ranges: "2012:2015" (inclusive of both 2012 and 2015)
    - Mixed: "2010,2012:2015,2017:2022,2024"

    Returns a list of years as strings.
    """
    if not year_filter_str:
        return []

    years = []

    # Split by comma
    parts = year_filter_str.split(",")

    for part in parts:
        if ":" in part:
            # Handle range
            start_year, end_year = part.split(":")
            try:
                start = int(start_year.strip())
                end = int(end_year.strip())

                if start > end:
                    logger.warning(
                        f"Invalid year range: {part} (start > end). Skipping."
                    )
                    continue

                # Add all years in the range (inclusive)
                years.extend([str(year) for year in range(start, end + 1)])
            except ValueError:
                logger.warning(f"Invalid year range: {part}. Skipping.")
                continue
        else:
            # Handle individual year
            try:
                # Make sure it's a valid year by parsing as integer
                int(part.strip())
                years.append(part.strip())
            except ValueError:
                logger.warning(f"Invalid year: {part}. Skipping.")
                continue

    return years


def safe_read_from_tar(tar_file, tar_member) -> Optional[str]:
    """
    Safely extract and read a file from a tar archive, handling potential errors.
    """
    try:
        f = tar_file.extractfile(tar_member)
        if not f:
            return None

        content = f.read()

        return content.decode("utf-8", errors="replace")

    except Exception as e:
        logger.warning(f"Error extracting {tar_member.name}: {e}")
        return None


def safe_csv_read(content: str) -> Tuple[List[str], List[List[str]]]:
    """
    Safely parse CSV content and handle potentially corrupt or malformed data.
    Properly handles quoted fields with internal commas.
    Returns (header_row, data_rows)
    """
    # Remove any NULL bytes that can cause CSV parsing errors
    content = content.replace("\0", "")

    if not content.strip():
        return [], []

    lines = content.splitlines()
    if len(lines) <= 1:
        return [], []

    try:
        reader = csv.reader(StringIO(content), quotechar='"', quoting=csv.QUOTE_MINIMAL)
        rows = list(reader)

        if not rows:
            return [], []

        header = rows[0]
        data_rows = rows[1:]

        # Check if we have rows with more fields than the header (only log once)
        if data_rows:
            max_fields = max(len(row) for row in data_rows)
            if max_fields > len(header):
                # Log just once with a summary rather than for each row
                logger.debug(
                    f"Some rows have more fields ({max_fields}) than header ({len(header)}). "
                )

        return header, data_rows

    except Exception as e:
        logger.warning(f"Error parsing CSV: {e}")
        return [], []


def check_system_resources():
    """
    Check available system resources and provide recommendations.
    Returns a dictionary with resource information.
    """
    resources = {
        "system": platform.system(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "ram_recommendation": "4-8GB",
    }

    mem = psutil.virtual_memory()
    resources["total_memory"] = mem.total / (1024 * 1024 * 1024)  # GB
    resources["available_memory"] = mem.available / (1024 * 1024 * 1024)  # GB

    # Provide recommendations based on available memory
    if resources["available_memory"] < 2:
        logger.warning("Low memory detected! Consider reducing max_workers")
        resources["ram_warning"] = True
    else:
        resources["ram_warning"] = False

    # Check disk space
    try:
        disk_usage = shutil.disk_usage(Path.cwd())
        resources["free_disk_space"] = disk_usage.free / (1024 * 1024 * 1024)  # GB

        if resources["free_disk_space"] < 10:
            logger.warning(
                "Low disk space detected! You may not have enough space for all archives"
            )
            resources["disk_warning"] = True
        else:
            resources["disk_warning"] = False
    except Exception as e:
        logger.info(f"Could not check disk space: {e}")
        resources["disk_check"] = False

    return resources
