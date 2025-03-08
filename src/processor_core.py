#!/usr/bin/env python3
"""
Core functions for processing NOAA weather data archives.
Implements the main orchestration logic with improved parallelism.
"""
import logging
import tarfile
import concurrent.futures
import gzip
import random
import gc
import io
import csv
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Dict, List, Set, Tuple, Optional
import polars as pl
from tqdm import tqdm


from config import MERGED_DIR, IO_BUFFER_SIZE, DEFAULT_MAX_WORKERS
from file_io import get_station_modified_dates, write_station_data_to_disk
from verification import should_process_station_file
from csv_merger import filter_csv_data
from metadata_manager import flush_metadata_updates, update_metadata_with_counts
from utils import safe_read_from_tar

logger = logging.getLogger("weather_processor")


def merge_station_data(
    all_archives: List[Path], max_workers: int = DEFAULT_MAX_WORKERS
) -> Tuple[Set[str], Dict[str, Dict[str, int]]]:
    """
    Process all archives and merge data by station ID.
    Archives are processed sequentially, but files within each archive are processed in parallel.
    Memory-optimized with batch metadata writing for better performance.

    Args:
        all_archives: List of paths to tar.gz archives
        max_workers: Maximum number of concurrent workers for processing files within an archive

    Returns:
        Tuple of (updated_stations, observation_counts)
        - updated_stations: Set of station IDs that were updated
        - observation_counts: Dictionary mapping station_id -> year -> count
    """

    # Get last modified dates for existing station files
    last_modified_dates = get_station_modified_dates()
    logger.info(f"Found {len(last_modified_dates)} existing station data files")

    # Track updated stations and observation counts
    updated_stations = set()
    observation_counts = defaultdict(lambda: defaultdict(int))

    # Process archives sequentially
    for archive in tqdm(all_archives, desc="Processing archives", unit="archive"):
        logger.info(f"Processing archive: {archive}")
        year = archive.stem

        try:
            # Extract list of station files from archive first
            station_files = []
            with tarfile.open(archive, "r:gz", bufsize=IO_BUFFER_SIZE) as tar:
                station_files = [m for m in tar.getmembers() if m.name.endswith(".csv")]

            total_station_files = len(station_files)
            logger.info(f"Found {total_station_files} station files in {archive}")

            # Shared variables to track updated stations and counts across all chunks
            all_updated_stations = set()
            archive_counts = defaultdict(int)
            stations_lock = Lock()

            # Function to process a chunk and write it directly to disk
            def process_and_write_chunk(
                archive_path, station_chunk, progress_bar, progress_lock, chunk_id
            ):
                try:
                    chunk_data, chunk_counts = process_station_chunk(
                        archive_path, station_chunk, progress_bar, progress_lock
                    )
                    chunk_updated = set(chunk_data.keys())

                    logger.debug(
                        f"Writing data for chunk {chunk_id} with {len(chunk_data)} stations"
                    )
                    write_station_data_to_disk(chunk_data)

                    # Update the global set of updated stations and observation counts
                    with stations_lock:
                        all_updated_stations.update(chunk_updated)
                        for station_id, count in chunk_counts.items():
                            archive_counts[station_id] = count

                    # Clear chunk data to free memory
                    chunk_data.clear()
                    gc.collect()

                    return len(chunk_updated)
                except Exception as e:
                    logger.error(
                        f"Error processing chunk {chunk_id}: {e}", exc_info=True
                    )
                    return 0

            # Process station files in parallel
            if max_workers > 1 and station_files:
                # Create a shared progress bar for all workers
                progress_lock = Lock()
                progress = tqdm(
                    total=total_station_files, desc=f"Processing {year}", unit="station"
                )

                base_chunk_size = max(1, total_station_files // (max_workers * 3))
                station_chunks = []
                start_idx = 0

                while start_idx < len(station_files):
                    # Vary chunk size by ±15% to reduce simultaneous writes
                    variation = random.uniform(0.85, 1.15)
                    chunk_size = max(1, int(base_chunk_size * variation))

                    end_idx = min(start_idx + chunk_size, len(station_files))
                    station_chunks.append(station_files[start_idx:end_idx])
                    start_idx = end_idx

                logger.info(
                    f"Created {len(station_chunks)} varied-size chunks with ~{base_chunk_size} stations each"
                )

                # Use ThreadPoolExecutor for concurrency within single archive
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    futures = [
                        executor.submit(
                            process_and_write_chunk,
                            archive,
                            chunk,
                            progress,
                            progress_lock,
                            i,
                        )
                        for i, chunk in enumerate(station_chunks)
                    ]

                    # Wait for all futures to complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            stations_processed = future.result()
                            logger.debug(
                                f"Chunk completed with {stations_processed} stations processed"
                            )
                        except Exception as e:
                            logger.error(f"Error in future: {e}", exc_info=True)

                progress.close()

                # Update the main set of updated stations
                updated_stations.update(all_updated_stations)
                logger.info(
                    f"Updated {len(all_updated_stations)} stations in this archive"
                )

            else:
                progress = tqdm(
                    total=total_station_files, desc=f"Processing {year}", unit="station"
                )

                # Process files in smaller chunks even in sequential mode
                chunk_size = min(500, total_station_files)
                for i in range(0, total_station_files, chunk_size):
                    chunk = station_files[i : i + chunk_size]
                    chunk_data, observation_counts = process_station_chunk(
                        archive, chunk, progress
                    )

                    # Write this chunk to disk
                    logger.debug(
                        f"Writing sequential chunk {i//chunk_size} with {len(chunk_data)} stations"
                    )
                    write_station_data_to_disk(chunk_data)

                    # Update tracked stations
                    updated_stations.update(chunk_data.keys())

                    # Clear memory
                    chunk_data.clear()
                    gc.collect()

                progress.close()

            # After processing the entire archive, flush metadata updates
            for station_id, count in archive_counts.items():
                if count > 0:
                    observation_counts[station_id][year] = count

            # Flush metadata updates to disk
            logger.info("Flushing metadata updates to disk...")
            num_updated = flush_metadata_updates()
            logger.info(f"Metadata updates completed for {num_updated} stations")

            # Update metadata with observation counts
            if archive_counts:
                logger.info(
                    f"Updating metadata with {len(archive_counts)} station observation counts for year {year}"
                )
                year_counts = {
                    station_id: {year: count}
                    for station_id, count in archive_counts.items()
                }
                update_metadata_with_counts(year_counts)

            # Force garbage collection after processing each archive
            gc.collect()

            # Update the main set of updated stations
            updated_stations.update(all_updated_stations)

        except Exception as e:
            logger.error(f"Error processing archive {archive}: {e}", exc_info=True)
            # Still try to flush metadata even if there was an error
            flush_metadata_updates()

    logger.info(f"Total updated data for {len(updated_stations)} stations")
    return updated_stations, observation_counts


def sort_station_files_chronologically():
    """
    Sort records in each station file from oldest to newest.
    Uses multiple threads and Polars for optimal performance.
    Requires Polars to be installed - no fallback method provided.
    """
    station_files = list(MERGED_DIR.glob("*.csv.gz"))
    total_files = len(station_files)

    logger.info(
        f"Sorting {total_files} station files chronologically using up to {DEFAULT_MAX_WORKERS} workers"
    )

    # Progress tracking
    progress = tqdm(total=total_files, desc="Sorting files", unit="file")
    progress_lock = Lock()

    # Results tracking
    success_count = 0
    failed_count = 0
    results_lock = Lock()

    def sort_station_file(station_path):
        """
        Sort a single station file chronologically using Polars.
        DATE is expected to be the second column (index 1) in ISO format.
        Treats all columns as strings except the date column.
        """
        try:
            # Read the gzipped CSV file
            with gzip.open(station_path, "rt", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # Split header and data to preserve header exactly
            if "\n" not in content:
                logger.warning(f"No data rows in {station_path.name}, skipping")
                return True

            header_line, data_content = content.split("\n", 1)

            # Parse CSV with proper header handling
            csv_reader = csv.reader(io.StringIO(header_line))
            headers = next(csv_reader, None)

            if not headers or len(headers) < 2:
                logger.warning(f"Invalid headers in {station_path.name}, skipping")
                return False

            # Confirm which column is the date column (should be 2nd column, index 1)
            date_col_idx = 1  # Assume DATE is second column
            date_col_name = (
                headers[date_col_idx] if len(headers) > date_col_idx else "DATE"
            )

            # Create schema with all columns as strings
            schema = {col: pl.Utf8 for col in headers}

            # Only parse the date column as datetime - keep everything else as strings
            if date_col_name in headers:
                schema[date_col_name] = pl.Datetime

            # Create a DataFrame with the CSV data - explicitly set schema
            df = pl.read_csv(
                io.StringIO(header_line + "\n" + data_content),
                schema=schema,
                try_parse_dates=False,  # Don't auto-parse dates, rely on schema
                infer_schema_length=0,  # Don't infer schema
            )

            # Drop duplicates to clean the data
            df = df.unique()

            # Sort by the date column
            if date_col_name in df.columns:
                sorted_df = df.sort(date_col_name)

                # Write back to file
                with gzip.open(station_path, "wt", encoding="utf-8") as f:
                    sorted_df.write_csv(f)

                return True
            else:
                logger.warning(
                    f"Could not find expected date column '{date_col_name}' in {station_path.name}"
                )
                return False

        except Exception as e:
            logger.error(f"Error sorting {station_path.name}: {e}")
            return False

    def process_file_with_progress(file_path):
        """Process a single file and update progress"""
        result = sort_station_file(file_path)

        # Update progress safely
        with progress_lock:
            progress.update(1)

        # Update results safely
        nonlocal success_count, failed_count
        with results_lock:
            if result:
                success_count += 1
            else:
                failed_count += 1

        return result

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=DEFAULT_MAX_WORKERS
    ) as executor:
        # Submit all files for processing
        futures = [
            executor.submit(process_file_with_progress, file_path)
            for file_path in station_files
        ]

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    # Close progress bar
    progress.close()

    # Log results
    logger.info(
        f"Completed sorting {total_files} station files: {success_count} succeeded, {failed_count} failed"
    )
    return success_count > 0


def process_station_file(
    tar_member, tar_file, station_id: str
) -> Tuple[bool, Optional[str], Optional[Dict[str, str]], int]:
    """
    Process a single station file from the tar archive.
    Uses timestamp verification to decide whether processing is needed.

    Returns (updated_flag, filtered_content, metadata, observation_count)
    """

    # Extract and read the file content
    content = safe_read_from_tar(tar_file, tar_member)
    if not content:
        return False, None, None, 0

    try:
        # Check if we should process this file based on timestamps
        if not should_process_station_file(station_id, content, tar_member.mtime):
            return False, None, None, 0

        # Count observations (number of data rows)
        count = 0
        if content:
            lines = content.splitlines()
            count = max(0, len(lines) - 1)  # Subtract 1 for header

        # Filter the content to separate metadata and weather data
        filtered_content, metadata = filter_csv_data(content)

        # Ensure station ID is in metadata
        metadata["STATION"] = station_id

        return bool(filtered_content), filtered_content, metadata, count

    except Exception as e:
        logger.error(f"Error processing {tar_member.name}: {e}", exc_info=True)
        return False, None, None, 0


def process_station_chunk(
    archive_path, station_members, progress_bar=None, progress_lock=None
):
    """
    Process a chunk of station files from a single archive.

    Args:
        archive_path: Path to the tar.gz archive
        station_members: List of tar members (files) to process
        progress_bar: Optional tqdm progress bar to update
        progress_lock: Thread lock for updating progress bar safely

    Returns:
        Tuple of (chunk_data, observation_counts)
        - chunk_data: Dictionary mapping station_id to (content, metadata, is_new)
        - observation_counts: Dictionary mapping station_id to observation count
    """
    chunk_data = {}
    observation_counts = {}
    year = Path(archive_path).stem

    try:
        with tarfile.open(archive_path, "r:gz", bufsize=IO_BUFFER_SIZE) as tar:
            for member in station_members:
                station_id = Path(member.name).stem

                try:
                    # Process the station file - note the added count return value
                    updated, content, metadata, count = process_station_file(
                        member, tar, station_id
                    )

                    if updated and content and metadata:
                        chunk_data[station_id] = (content, metadata, True)
                        observation_counts[station_id] = count

                except Exception as e:
                    logger.error(
                        f"Error processing station {station_id}: {e}", exc_info=True
                    )

                # Update progress bar if provided
                if progress_bar:
                    if progress_lock:
                        with progress_lock:
                            progress_bar.update(1)
                    else:
                        progress_bar.update(1)

    except Exception as e:
        logger.error(f"Error in chunk processing: {e}", exc_info=True)

    return chunk_data, observation_counts
