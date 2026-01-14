#!/usr/bin/env python3
"""
Core functions for processing NOAA weather data archives.
Implements the main orchestration logic with improved parallelism.
"""
import concurrent.futures
import gc
import gzip
import logging
import os
import queue
import shutil
import tarfile
import threading
import time
import csv
import io
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from threading import Lock
from tqdm import tqdm
import polars as pl

from config import (
    DEFAULT_MAX_WORKERS,
    EXTRACTION_WORKER_NUMBER,
    IO_BUFFER_SIZE,
    MERGED_DIR,
    TEMP_DIR,
    PROGRESS_LOG_INTERVAL,
    QUEUE_SIZE,
)
from csv_merger import filter_csv_data, merge_csv_data_with_polars
from file_io import get_station_modified_dates
from metadata_manager import (
    update_station_metadata,
    update_metadata_with_counts,
    finalize_metadata,
)
from utils import safe_read_from_tar
from verification import should_process_station_file

logger = logging.getLogger("weather_processor")


class StationProcessor:
    """
    Manages the producer/consumer pattern for processing station data from archives.
    """

    def __init__(
        self,
        archive_path: Path,
        max_extraction_workers: int = 1,
        max_processing_workers: int = 6,
        queue_size: int = QUEUE_SIZE,
    ):
        self.archive_path = archive_path
        self.max_extraction_workers = max_extraction_workers
        self.max_processing_workers = max_processing_workers
        self.queue_size = queue_size

        # Extract just the year (remove .tar from year.tar.gz)
        self.year = archive_path.stem.split(".")[0]

        # Create temp directory under ../data/temp
        temp_root = TEMP_DIR
        temp_root.mkdir(parents=True, exist_ok=True)

        # Create a unique subdirectory for this process
        process_temp_dir = f"process_{os.getpid()}_{int(time.time())}"
        self.temp_dir = temp_root / process_temp_dir
        self.temp_dir.mkdir(exist_ok=True)

        # Store temp root for cleanup
        self.temp_root = temp_root

        # Communication mechanisms
        self.file_queue = queue.Queue(maxsize=queue_size)
        self.extraction_complete = threading.Event()

        # Thread tracking for safe cleanup
        self.active_threads = set()
        self.threads_lock = threading.Lock()
        self.safe_to_cleanup = threading.Event()

        # Results tracking
        self.updated_stations = set()
        self.observation_counts = {}
        self.results_lock = Lock()

        # Station modification date cache
        self.station_dates = get_station_modified_dates()

        # Progress tracking
        self.total_files = 0
        self.processed_count = 0
        self.processed_lock = Lock()

    def register_thread(self, thread):
        """Register a thread for tracking"""
        with self.threads_lock:
            self.active_threads.add(thread.ident)

    def unregister_thread(self, thread_id=None):
        """Unregister a thread when it's done"""
        thread_id = thread_id or threading.current_thread().ident
        with self.threads_lock:
            if thread_id in self.active_threads:
                self.active_threads.remove(thread_id)
            # If no more active threads, signal it's safe to clean up
            if not self.active_threads:
                self.safe_to_cleanup.set()

    def cleanup(self):
        """Clean up temporary files and directory"""
        try:
            # Wait for all threads to signal they're done
            if not self.safe_to_cleanup.is_set():
                logger.info("Waiting for all threads to complete before cleanup...")
                # Wait indefinitely - since threads are not daemon threads,
                # they will complete their work even if this takes a while
                self.safe_to_cleanup.wait()
                logger.info("All threads have completed, proceeding with cleanup")

            # Give a small delay to ensure all file operations have completed
            time.sleep(0.5)

            if self.temp_dir.exists():
                # Try several times to clean up files in case they're still being used
                max_attempts = 3
                for attempt in range(max_attempts):
                    success = True
                    for file in self.temp_dir.iterdir():
                        try:
                            file.unlink()
                        except Exception as e:
                            success = False
                            if attempt == max_attempts - 1:  # Only log on last attempt
                                logger.warning(f"Error removing temp file {file}: {e}")

                    if success:  # If all files were removed successfully
                        break

                    # Wait a bit before trying again
                    time.sleep(1)

                # Try to remove the directory
                try:
                    if self.temp_dir.exists():
                        try:
                            # Try with normal rmdir
                            self.temp_dir.rmdir()
                            logger.debug(f"Removed temporary directory {self.temp_dir}")
                        except OSError:
                            # If that fails, try with stronger methods
                            try:
                                import shutil

                                shutil.rmtree(self.temp_dir, ignore_errors=True)
                                logger.debug(
                                    f"Removed temporary directory with rmtree {self.temp_dir}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Could not remove temp dir {self.temp_dir} with rmtree: {e}"
                                )
                except Exception as e:
                    logger.warning(f"Error removing temp dir {self.temp_dir}: {e}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def producer_task(self, member_batch):
        """
        Extract files from the archive and add to the queue.

        Args:
            member_batch: List of tar members to extract
        """
        try:
            # Register this thread
            self.register_thread(threading.current_thread())

            logger.debug(f"Producer starting with {len(member_batch)} files")
            with tarfile.open(self.archive_path, "r:gz", bufsize=IO_BUFFER_SIZE) as tar:
                for member in member_batch:
                    try:
                        # Extract the file to temp dir
                        extraction_path = self.temp_dir / Path(member.name).name
                        tar.extract(member, path=self.temp_dir)

                        # Add to queue
                        self.file_queue.put(
                            (extraction_path, member.name, member.mtime)
                        )

                    except Exception as e:
                        logger.error(f"Error extracting {member.name}: {e}")
                        # Skip this file but continue with others

            logger.debug("Producer completed batch extraction")

        except Exception as e:
            logger.error(f"Producer thread error: {e}", exc_info=True)
        finally:
            # Unregister this thread when done
            self.unregister_thread()

    def process_station_content(
        self, content: str, station_id: str, mtime: float
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, str]], int]:
        """
        Process station content.

        Args:
            content: The CSV content as a string
            station_id: The station identifier
            mtime: Modification time from the archive

        Returns:
            Tuple of (updated_flag, filtered_content, metadata, observation_count)
        """
        try:
            # Check if we should process this file based on timestamps
            if not should_process_station_file(station_id, content, mtime):
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
            logger.error(
                f"Error processing content for {station_id}: {e}", exc_info=True
            )
            return False, None, None, 0

    def write_station_data(
        self, station_id: str, content: str, metadata: Dict[str, str], count: int
    ):
        """
        Write station data to disk with proper merging.
        Optimized to use Polars directly for reading/writing, eliminating redundant string operations.

        Args:
            station_id: The station identifier
            content: Filtered CSV content to write
            metadata: Station metadata
            count: Observation count
        """
        try:
            # Prepare path
            station_path = MERGED_DIR / f"{station_id}.csv.gz"

            # Append or create station data file
            if station_path.exists():
                try:
                    # Read existing file directly with Polars (no intermediate string)
                    # Polars can read from gzip files directly by path, or from file handles
                    try:
                        # Try reading directly from gzip file handle
                        with gzip.open(station_path, "rb") as f:
                            existing_df = pl.read_csv(
                                f,
                                has_header=True,
                                quote_char='"',
                                ignore_errors=True,
                                infer_schema_length=0,
                                try_parse_dates=False,
                            )
                        # Verify we got data
                        if existing_df.shape[0] == 0:
                            raise ValueError("Empty DataFrame read from file")
                    except Exception as read_e:
                        # Fallback: read as string if direct read fails
                        logger.debug(f"Direct Polars read failed, using string fallback: {read_e}")
                        with gzip.open(
                            station_path, "rt", encoding="utf-8", errors="replace"
                        ) as f:
                            existing_content = f.read()
                        merged_content = merge_csv_data_with_polars(
                            existing_content, content
                        )
                        with gzip.open(station_path, "wt", encoding="utf-8") as f:
                            f.write(merged_content)
                        update_station_metadata(station_id, metadata)
                        return

                    # Parse new content
                    from io import StringIO
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
                    from csv_merger import get_canonical_headers
                    canonical_headers = get_canonical_headers(merged_df.columns)
                    merged_df = merged_df.select([pl.col(col) for col in canonical_headers])

                    # Format DATE column back to string if present
                    if "DATE" in merged_df.columns:
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
                except Exception as e:
                    logger.error(f"Error creating new station file {station_id}: {e}")

            # Update metadata - accumulate changes in memory
            update_station_metadata(station_id, metadata)

        except Exception as e:
            logger.error(
                f"Error writing station data for {station_id}: {e}", exc_info=True
            )

    def consumer_task(self, pbar=None):
        """Process files from the queue until extraction is complete and queue is empty"""
        try:
            # Register this thread
            self.register_thread(threading.current_thread())

            logger.debug("Consumer starting")
            while not (self.extraction_complete.is_set() and self.file_queue.empty()):
                try:
                    # Get a file with timeout to periodically check if extraction is complete
                    try:
                        file_path, original_name, mtime = self.file_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        continue

                    # Process the file
                    station_id = Path(original_name).stem

                    try:
                        # Read the extracted file
                        with open(
                            file_path, "r", encoding="utf-8", errors="replace"
                        ) as f:
                            content = f.read()

                        # Process the station data
                        updated, filtered_content, metadata, count = (
                            self.process_station_content(content, station_id, mtime)
                        )

                        if updated and filtered_content and metadata:
                            self.write_station_data(
                                station_id, filtered_content, metadata, count
                            )

                            with self.results_lock:
                                self.updated_stations.add(station_id)
                                if count > 0:
                                    self.observation_counts[station_id] = count

                    except Exception as e:
                        logger.error(f"Error processing {station_id}: {e}")

                    finally:
                        # Update progress counter
                        with self.processed_lock:
                            self.processed_count += 1
                            if (
                                self.processed_count % PROGRESS_LOG_INTERVAL == 0
                                or self.processed_count == self.total_files
                            ):
                                logger.info(
                                    f"Processed {self.processed_count}/{self.total_files} files from {self.year}"
                                )

                        # Update progress bar if provided
                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix({"updated": len(self.updated_stations)})

                        # Remove temp file and mark task as done
                        try:
                            if file_path.exists():
                                file_path.unlink()
                        except Exception as e:
                            logger.warning(f"Error removing temp file {file_path}: {e}")

                        self.file_queue.task_done()

                except Exception as e:
                    logger.error(f"Error in consumer task: {e}")

            logger.debug("Consumer finished")

        except Exception as e:
            logger.error(f"Consumer thread error: {e}", exc_info=True)
        finally:
            # Unregister this thread when done
            self.unregister_thread()

    def process(self):
        """
        Process the archive using the producer/consumer pattern.

        Returns:
            Tuple of (updated_stations, observation_counts)
        """
        try:
            year = self.archive_path.stem
            logger.info(
                f"Starting producer/consumer processing for {self.archive_path}"
            )

            # Get list of all members first
            logger.debug(f"Scanning archive {self.archive_path} for CSV files")
            start_time = time.time()

            try:
                with tarfile.open(
                    self.archive_path, "r:gz", bufsize=IO_BUFFER_SIZE
                ) as tar:
                    members = [m for m in tar.getmembers() if m.name.endswith(".csv")]
                    self.total_files = len(members)
            except Exception as e:
                logger.error(f"Error scanning archive {self.archive_path}: {e}")
                return set(), {}

            logger.info(f"Found {self.total_files} files in {self.archive_path}")
            logger.debug(f"Archive scan completed in {time.time() - start_time:.2f}s")

            # Initialize progress bar
            pbar = tqdm(
                total=self.total_files,
                desc=f"Processing {year}",
                unit="files",
                leave=True,
                position=0,
            )

            # Start consumer threads - IMPORTANT: Not setting as daemon threads
            logger.debug(f"Starting {self.max_processing_workers} consumer threads")
            consumers = []
            for i in range(self.max_processing_workers):
                consumer = threading.Thread(target=self.consumer_task, args=(pbar,))
                # NOT setting daemon=True - we want these to finish their work
                consumer.start()
                consumers.append(consumer)

            # Split members into batches for producers
            batch_size = max(1, len(members) // self.max_extraction_workers)
            member_batches = [
                members[i : i + batch_size] for i in range(0, len(members), batch_size)
            ]

            # Start producer threads - IMPORTANT: Not setting as daemon threads
            logger.debug(f"Starting {len(member_batches)} producer threads")
            producers = []
            for i, batch in enumerate(member_batches):
                producer = threading.Thread(target=self.producer_task, args=(batch,))
                # NOT setting daemon=True - we want these to finish their work
                producer.start()
                producers.append(producer)

            # Wait for producers to finish WITHOUT timeout
            logger.debug("Waiting for producer threads to complete")
            for producer in producers:
                try:
                    producer.join()  # No timeout - wait as long as needed
                except Exception as e:
                    logger.error(f"Error joining producer thread: {e}")

            # Signal extraction is complete
            logger.debug("All producers completed, marking extraction as complete")
            self.extraction_complete.set()

            # Wait for the queue to be empty
            logger.debug("Waiting for queue to be empty")
            while not self.file_queue.empty():
                logger.debug(
                    f"Queue still has {self.file_queue.qsize()} items, waiting..."
                )
                time.sleep(5)  # Check every 5 seconds

            # Now wait for consumer threads WITHOUT timeout
            logger.debug("Queue is empty, waiting for consumer threads to complete")
            for consumer in consumers:
                try:
                    consumer.join()  # No timeout - wait as long as needed
                except Exception as e:
                    logger.error(f"Error joining consumer thread: {e}")

            # Now it's definitely safe to clean up
            self.safe_to_cleanup.set()

            # Close progress bar
            pbar.close()

            # Explicitly check if we've processed all files
            if self.processed_count < self.total_files:
                logger.warning(
                    f"Not all files were processed: {self.processed_count}/{self.total_files}"
                )

            # Process is complete
            logger.info(
                f"Processed {self.processed_count}/{self.total_files} files from {self.archive_path}"
            )

            # Update metadata with observation counts for this year
            if self.observation_counts:
                try:
                    logger.info(
                        f"Updating metadata with {len(self.observation_counts)} station counts for year {self.year}"
                    )
                    year_counts = {
                        station_id: {self.year: count}
                        for station_id, count in self.observation_counts.items()
                    }
                    update_metadata_with_counts(year_counts)
                    logger.info(
                        f"All metadata updates for {len(self.updated_stations)} stations committed"
                    )
                except Exception as e:
                    logger.error(f"Error updating metadata: {e}", exc_info=True)
            else:
                logger.info(f"No observation counts to update for year {self.year}")
                logger.info(
                    f"All metadata updates for {len(self.updated_stations)} stations committed"
                )

            return self.updated_stations, self.observation_counts

        except Exception as e:
            logger.error(f"Error in producer/consumer processing: {e}", exc_info=True)
            return set(), {}

        finally:
            # Always clean up temp files, but now it will wait for threads to finish
            self.cleanup()


# In processor_core.py, modify the merge_station_data function:
def merge_station_data(
    all_archives: List[Path], max_workers: int = DEFAULT_MAX_WORKERS
) -> Tuple[Set[str], Dict[str, Dict[str, int]]]:
    """
    Process all archives and merge data by station ID using producer/consumer pattern.
    Archives are processed sequentially, but files within archive are processed concurrently.

    Args:
        all_archives: List of paths to tar.gz archives
        max_workers: Maximum number of workers to use for processing

    Returns:
        Tuple of (updated_stations, observation_counts)
        - updated_stations: Set of station IDs that were updated
        - observation_counts: Dictionary mapping station_id -> year -> count
    """
    # Use fixed extraction workers and rest for processing
    extraction_workers = EXTRACTION_WORKER_NUMBER
    processing_workers = max(1, max_workers - extraction_workers)

    logger.info(
        f"Using {extraction_workers} extraction workers and {processing_workers} processing workers"
    )

    # Track all updated stations and observation counts
    all_updated_stations = set()
    all_observation_counts = {}

    # Process archives sequentially
    for archive in all_archives:
        start_time = time.time()
        logger.info(f"Processing archive: {archive}")

        try:
            # Create processor for this archive
            processor = StationProcessor(
                archive_path=archive,
                max_extraction_workers=extraction_workers,
                max_processing_workers=processing_workers,
                queue_size=QUEUE_SIZE,
            )

            # Process the archive
            updated_stations, observation_counts = processor.process()

            # Update overall tracking
            all_updated_stations.update(updated_stations)
            for station_id, count in observation_counts.items():
                if station_id not in all_observation_counts:
                    all_observation_counts[station_id] = {}
                all_observation_counts[station_id][archive.stem.split(".")[0]] = count

            # Log results for this archive
            elapsed_time = time.time() - start_time
            logger.info(f"Completed processing {archive} in {elapsed_time:.2f}s")
            logger.info(f"Updated {len(updated_stations)} stations in {archive}")

            # Export current metadata to CSV after each archive for incremental backup
            try:
                logger.info(
                    f"Exporting intermediate metadata to CSV after processing {archive}"
                )
                from metadata_manager import export_to_csv

                export_to_csv()
            except Exception as e:
                logger.error(
                    f"Error exporting intermediate metadata to CSV: {e}", exc_info=True
                )

            # Force garbage collection
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing archive {archive}: {e}", exc_info=True)
            # Continue with next archive

    # Finalize metadata and export to CSV
    logger.info("Finalizing metadata and exporting to CSV")
    finalize_metadata()

    logger.info(f"Total updated data for {len(all_updated_stations)} stations")
    return all_updated_stations, all_observation_counts


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
        Optimized to read directly from gzip file without intermediate string conversion.
        DATE is expected to be the second column (index 1) in ISO format.
        Treats all columns as strings except the date column.
        """
        try:
            # Read directly from gzipped CSV file with Polars
            try:
                with gzip.open(station_path, "rb") as f:
                    # Read first to get headers
                    df = pl.read_csv(
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
                with gzip.open(station_path, "rt", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                if "\n" not in content:
                    logger.warning(f"No data rows in {station_path.name}, skipping")
                    return True

                header_line, data_content = content.split("\n", 1)
                csv_reader = csv.reader(io.StringIO(header_line))
                headers = next(csv_reader, None)

                if not headers or len(headers) < 2:
                    logger.warning(f"Invalid headers in {station_path.name}, skipping")
                    return False

                date_col_name = headers[1] if len(headers) > 1 else "DATE"
                schema = {col: pl.Utf8 for col in headers}
                if date_col_name in headers:
                    schema[date_col_name] = pl.Datetime

                df = pl.read_csv(
                    io.StringIO(header_line + "\n" + data_content),
                    schema=schema,
                    try_parse_dates=False,
                    infer_schema_length=0,
                )

            # Check if we have data
            if df.shape[0] == 0:
                logger.warning(f"No data rows in {station_path.name}, skipping")
                return True

            # Find date column (should be second column or named DATE)
            date_col_name = None
            if len(df.columns) > 1:
                # Try second column first
                potential_date_col = df.columns[1]
                if "DATE" in potential_date_col.upper() or potential_date_col == "DATE":
                    date_col_name = potential_date_col
            if not date_col_name and "DATE" in df.columns:
                date_col_name = "DATE"

            if not date_col_name:
                logger.warning(
                    f"Could not find date column in {station_path.name}, skipping sort"
                )
                return False

            # Parse date column if it's a string
            if df[date_col_name].dtype == pl.Utf8:
                df = df.with_columns(
                    [pl.col(date_col_name).str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False)]
                )

            # Drop duplicates and sort
            df = df.unique().sort(date_col_name)

            # Format date back to string if needed
            if df[date_col_name].dtype == pl.Datetime:
                df = df.with_columns(
                    [pl.col(date_col_name).dt.strftime("%Y-%m-%dT%H:%M:%S")]
                )

            # Cast all to strings for consistent output
            df = df.select([pl.col(col).cast(pl.Utf8) for col in df.columns])

            # Write back to file directly
            with gzip.open(station_path, "wb") as f:
                df.write_csv(f, quote_style="necessary")

            return True

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
