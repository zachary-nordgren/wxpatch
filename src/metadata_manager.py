#!/usr/bin/env python3
"""
Functions for managing station metadata in the weather data processor.
"""
import csv
import uuid
import logging
import shutil
import tarfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from tqdm import tqdm

from config import MERGED_DIR, METADATA_FIELDS, METADATA_BATCH_SIZE
from utils import safe_read_from_tar

logger = logging.getLogger("weather_processor")


def load_metadata_file() -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Load existing metadata file.
    Returns (headers, metadata_dict)
    """
    metadata_path = MERGED_DIR / "wx_info.csv"
    headers = METADATA_FIELDS
    metadata_dict = {}

    if metadata_path.exists():
        try:
            with open(
                metadata_path, "r", encoding="utf-8", newline="", errors="replace"
            ) as f:
                reader = csv.reader(f)
                headers = next(reader, METADATA_FIELDS)

                for row in reader:
                    if row and len(row) > 0:
                        station_id = row[0]
                        # Pad row if it's shorter than headers
                        while len(row) < len(headers):
                            row.append("")
                        metadata_dict[station_id] = row
        except Exception as e:
            logger.error(f"Error reading metadata file: {e}", exc_info=True)
            # Create a backup of the corrupted file
            backup_path = metadata_path.with_suffix(".csv.bak")
            if metadata_path.exists():
                shutil.copy(metadata_path, backup_path)
                logger.info(
                    f"Created backup of corrupted metadata file at {backup_path}"
                )

    return headers, metadata_dict


def save_metadata_file(headers: List[str], metadata_dict: Dict[str, List[str]]):
    """
    Save metadata to file with error handling.
    """
    metadata_path = MERGED_DIR / "wx_info.csv"

    # Create a temporary file first
    temp_path = MERGED_DIR / f"wx_info_temp_{uuid.uuid4()}.csv"

    try:
        with open(temp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(headers)

            for station_id in sorted(metadata_dict.keys()):
                row = metadata_dict[station_id]
                # Ensure row has same length as headers
                while len(row) < len(headers):
                    row.append("")
                writer.writerow(row)

        # If we get here, writing was successful, replace the old file
        if metadata_path.exists():
            metadata_path.unlink()
        temp_path.rename(metadata_path)

    except Exception as e:
        logger.error(f"Error saving metadata file: {e}", exc_info=True)
        if temp_path.exists():
            temp_path.unlink()


def update_metadata_file(
    station_id: str, metadata: Dict[str, str], write_to_disk=False
):
    """
    Update the wx_info.csv metadata file with station information.
    Only updates if the station doesn't exist or if metadata has changed.
    By default, only accumulates updates in memory without writing to disk.

    Args:
        station_id: Station ID to update
        metadata: Dictionary of metadata values
        write_to_disk: Whether to write changes to disk immediately
    """
    # Use a static variable to track all pending updates
    if not hasattr(update_metadata_file, "pending_updates"):
        update_metadata_file.pending_updates = {}
        update_metadata_file.headers = None
        update_metadata_file.metadata_dict = None

    # Add this update to pending
    update_metadata_file.pending_updates[station_id] = metadata

    # Load metadata if needed
    if (
        update_metadata_file.headers is None
        or update_metadata_file.metadata_dict is None
    ):
        headers, metadata_dict = load_metadata_file()
        update_metadata_file.headers = headers
        update_metadata_file.metadata_dict = metadata_dict

    # If requested to write immediately (legacy behavior), process and save
    if (
        write_to_disk
        and len(update_metadata_file.pending_updates) >= METADATA_BATCH_SIZE
    ):
        flush_metadata_updates()


def flush_metadata_updates():
    """
    Process all pending metadata updates and write to disk.
    Returns the number of stations updated.
    """
    if (
        not hasattr(update_metadata_file, "pending_updates")
        or not update_metadata_file.pending_updates
    ):
        return 0

    # Make sure metadata is loaded
    if (
        update_metadata_file.headers is None
        or update_metadata_file.metadata_dict is None
    ):
        headers, metadata_dict = load_metadata_file()
        update_metadata_file.headers = headers
        update_metadata_file.metadata_dict = metadata_dict
    else:
        headers = update_metadata_file.headers
        metadata_dict = update_metadata_file.metadata_dict

    # Track stations that were added or updated
    new_stations = []
    updated_stations = []

    # Process all pending updates
    for sid, meta in update_metadata_file.pending_updates.items():
        # Create row from metadata
        new_row = [meta.get(field, "") for field in METADATA_FIELDS]

        # Set station ID
        new_row[0] = sid

        # Check if update is needed
        if sid not in metadata_dict:
            metadata_dict[sid] = new_row
            new_stations.append(sid)
        else:
            existing_row = metadata_dict[sid]
            # Copy existing year counts (if any)
            for i in range(len(METADATA_FIELDS), min(len(existing_row), len(headers))):
                if i < len(headers) and i < len(existing_row):
                    if len(new_row) <= i:
                        new_row.append(existing_row[i])
                    else:
                        new_row[i] = existing_row[i]

            # Only update if metadata has changed
            metadata_changed = False
            for i in range(min(len(new_row), len(existing_row))):
                if (
                    i < len(METADATA_FIELDS)
                    and new_row[i] != existing_row[i]
                    and new_row[i]
                ):
                    metadata_changed = True
                    break

            if metadata_changed:
                metadata_dict[sid] = new_row
                updated_stations.append(sid)

    # Log summary messages and save if changes were made
    if new_stations or updated_stations:
        if new_stations:
            logger.info(f"Added {len(new_stations)} new stations to metadata")
            logger.debug(f"New stations: {', '.join(new_stations)}")

        if updated_stations:
            logger.info(f"Updated metadata for {len(updated_stations)} stations")
            logger.debug(f"Updated stations: {', '.join(updated_stations)}")

        save_metadata_file(headers, metadata_dict)

    # Clear pending updates
    update_count = len(new_stations) + len(updated_stations)
    update_metadata_file.pending_updates = {}

    return update_count


def count_observations_by_year(
    all_archives: List[Path], stations: Set[str] = None
) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    """
    Count observations for each station by year.
    Returns a dictionary mapping station IDs to a dictionary of year -> count.

    Args:
        all_archives: List of paths to tar.gz archives
        stations: Optional set of station IDs to count. If None, count all stations.
    """
    observations = defaultdict(lambda: defaultdict(int))
    years = set()

    # For each archive, count observations in each station file
    for archive in tqdm(
        all_archives, desc="Counting observations by year", unit="archive"
    ):
        year = archive.stem
        if not year.isdigit():
            continue

        years.add(year)
        logger.info(f"Counting observations in {year}")

        try:
            with tarfile.open(archive, "r:gz") as tar:
                # Get list of all CSV files in the archive
                members = [m for m in tar.getmembers() if m.name.endswith(".csv")]

                for member in members:
                    station_id = Path(member.name).stem

                    if not stations or station_id in stations:
                        # Count lines in the file minus the header
                        try:
                            content = safe_read_from_tar(tar, member)
                            if content:
                                lines = content.splitlines()
                                count = len(lines) - 1 if len(lines) > 0 else 0
                                observations[station_id][year] = count
                        except Exception as e:
                            logger.debug(
                                f"Error counting observations for {station_id} in {year}: {e}"
                            )

        except Exception as e:
            logger.error(f"Error reading archive {archive}: {e}")

    return observations, sorted(years)


def update_year_counts(all_archives: List[Path], updated_stations: Set[str] = None):
    """
    Update the year counts in the wx_info.csv file.

    Args:
        all_archives: List of paths to tar.gz archives
        updated_stations: Optional set of station IDs to update. If None, update all stations.
    """
    if updated_stations is not None and not updated_stations:
        logger.info("No stations updated, skipping year count update")
        return

    metadata_path = MERGED_DIR / "wx_info.csv"
    if not metadata_path.exists():
        logger.warning("Metadata file doesn't exist, cannot update year counts")
        return

    # Load existing metadata
    headers, metadata_dict = load_metadata_file()

    # Get the years from the archives
    archive_years = set()
    for archive in all_archives:
        year = archive.stem
        if year.isdigit():
            archive_years.add(year)

    # Create updated headers with years
    updated_headers = headers.copy()
    for year in sorted(archive_years):
        if year not in headers:
            updated_headers.append(year)

    # Get observation counts
    logger.info("Counting observations by year...")
    observations, years = count_observations_by_year(all_archives, updated_stations)

    # Update metadata with year counts
    updates_made = False
    for station_id, year_counts in observations.items():
        if station_id in metadata_dict:
            row = metadata_dict[station_id]

            # Extend row if needed
            while len(row) < len(updated_headers):
                row.append("")

            # Update year counts
            for year, count in year_counts.items():
                if year in updated_headers:
                    year_idx = updated_headers.index(year)
                    if year_idx >= len(row):
                        # Extend row to accommodate the year column
                        row.extend([""] * (year_idx - len(row) + 1))

                    # Only update if count is different
                    if str(row[year_idx]) != str(count):
                        row[year_idx] = str(count)
                        updates_made = True

            metadata_dict[station_id] = row

    # Write updated metadata only if changes were made
    if updates_made:
        logger.info("Saving updated station observation counts to metadata file")
        save_metadata_file(updated_headers, metadata_dict)
    else:
        logger.info("No changes to observation counts, metadata file not updated")


def recover_corrupted_metadata():
    """
    Attempt to recover corrupted metadata file by parsing it line by line.
    This function tries to rebuild the metadata file if it's corrupted.
    """
    metadata_path = MERGED_DIR / "wx_info.csv"
    if not metadata_path.exists():
        return False

    backup_path = metadata_path.with_suffix(".csv.bak")
    recovery_path = metadata_path.with_suffix(".csv.recovered")

    # Create a backup if it doesn't exist
    if not backup_path.exists():
        try:
            shutil.copy(metadata_path, backup_path)
            logger.info(f"Created backup of metadata file at {backup_path}")
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")

    try:
        # Read the file line by line and skip corrupt lines
        valid_lines = []
        headers = None

        with open(metadata_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                # Remove NULL bytes
                clean_line = line.replace("\0", "")
                if i == 0:
                    # This is the header line
                    headers = clean_line.strip()
                    valid_lines.append(headers)
                else:
                    # Check if this is a valid row
                    try:
                        # Simple check: does it have roughly the right number of commas?
                        comma_count = clean_line.count(",")
                        expected_commas = headers.count(",")

                        if (
                            comma_count >= expected_commas - 3
                            and comma_count <= expected_commas + 3
                        ):
                            valid_lines.append(clean_line.strip())
                    except Exception:
                        # Skip lines that cause parsing errors
                        continue

        # Write the valid lines to the recovery file
        with open(recovery_path, "w", encoding="utf-8") as f:
            for line in valid_lines:
                f.write(line + "\n")

        # Replace the original with the recovered version
        if recovery_path.exists():
            if metadata_path.exists():
                metadata_path.unlink()
            recovery_path.rename(metadata_path)
            logger.info(f"Recovered metadata file with {len(valid_lines)-1} stations")
            return True

    except Exception as e:
        logger.error(f"Failed to recover metadata file: {e}")

    return False


def update_metadata_with_counts(observation_counts: Dict[str, Dict[str, int]]):
    """
    Update metadata with observation counts without re-reading archives.

    Args:
        observation_counts: Dictionary mapping station_id -> year -> count
    """
    # Load existing metadata
    headers, metadata_dict = load_metadata_file()

    # Get all years from the counts
    years = set()
    for station_counts in observation_counts.values():
        years.update(station_counts.keys())

    # Create updated headers with years
    updated_headers = headers.copy()
    for year in sorted(years):
        if year not in updated_headers:
            updated_headers.append(year)

    # Update metadata with counts
    updates_made = False
    for station_id, year_counts in observation_counts.items():
        if station_id in metadata_dict:
            row = metadata_dict[station_id]

            # Extend row if needed
            while len(row) < len(updated_headers):
                row.append("")

            # Update year counts
            for year, count in year_counts.items():
                if year in updated_headers:
                    year_idx = updated_headers.index(year)
                    if year_idx >= len(row):
                        row.extend([""] * (year_idx - len(row) + 1))

                    # Only update if count is different or not set
                    current_count = row[year_idx] if year_idx < len(row) else ""
                    if not current_count or str(current_count) != str(count):
                        row[year_idx] = str(count)
                        updates_made = True

            metadata_dict[station_id] = row

    # Write updated metadata if changes were made
    if updates_made:
        logger.info("Saving updated station observation counts to metadata file")
        save_metadata_file(updated_headers, metadata_dict)
    else:
        logger.info("No changes to observation counts, metadata file not updated")
