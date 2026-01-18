#!/usr/bin/env python3
"""
Downloader for GHCNh (Global Historical Climatology Network hourly) parquet files.

Downloads individual station parquet files from NOAA's S3-style object store.
Files are organized by year and saved to data/raw/ghcnh/{year}/ directories.

Features:
- Automatic retries with exponential backoff
- Concurrent downloads with adaptive rate limiting
- Resume/restart capability (skips already downloaded files)
- Rich progress bars with bandwidth and ETA estimates
- State persistence for interrupted downloads

Usage:
    python ghcnh_downloader.py                     # Download all years, US, CA, MX stations only
    python ghcnh_downloader.py --year 2024         # Download specific year
    python ghcnh_downloader.py --year 2020:2024    # Download year range
    python ghcnh_downloader.py --all-stations      # Download all stations (not just US)
    python ghcnh_downloader.py --list-years        # List available years
    python ghcnh_downloader.py --status            # Show download status/progress
"""
import argparse
import json
import logging
import time
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from threading import Lock, Semaphore
from typing import List, Tuple, Optional, Generator
import requests
from tqdm import tqdm

from config import (
    DATA_DIR,
    DOWNLOAD_TIMEOUT,
    DOWNLOAD_CHUNK_SIZE,
    MAX_DOWNLOAD_RETRIES,
    MAX_DOWNLOAD_THREADS,
)

# GHCNh-specific configuration
GHCNH_BASE_URL = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/"
GHCNH_RAW_DIR = DATA_DIR / "raw" / "ghcnh"
STATE_FILE = GHCNH_RAW_DIR / ".download_state.json"

# North America station prefixes (FIPS 10-4 country codes)
# Includes US territories: RQ (Puerto Rico), VQ (US Virgin Islands)
NORTH_AMERICA_PREFIXES = ("US", "CA", "MX", "RQ", "VQ")

# Adaptive rate limiting settings
INITIAL_CONCURRENT = 8  # Start conservative
MAX_CONCURRENT = 12  # Maximum concurrent downloads
MIN_CONCURRENT = 2  # Minimum concurrent downloads
RATE_LIMIT_CODES = {429, 503, 504}  # HTTP codes indicating rate limiting
CONNECTION_RESET_ERRORS = (ConnectionResetError, ConnectionError, requests.exceptions.ConnectionError)

# Progress bar settings
TQDM_NCOLS = 100  # Fixed width to prevent line wrapping issues

# Log file path
LOG_FILE = Path(__file__).parent / "ghcnh_downloader.log"

logger = logging.getLogger("weather_processor")


def setup_logging(verbose: bool = False, quiet: bool = False):
    """Configure logging for standalone execution with both console and file output."""
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Get the root logger for this module
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels, handlers filter

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with user-specified level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler - always log INFO and above
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger.info(f"Logging to file: {LOG_FILE}")


@dataclass
class DownloadStats:
    """Track download statistics for progress and bandwidth estimation."""
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_bytes: int = 0
    downloaded_bytes: int = 0
    start_time: float = field(default_factory=time.time)
    recent_speeds: deque = field(default_factory=lambda: deque(maxlen=20))
    lock: Lock = field(default_factory=Lock)

    def add_download(self, bytes_downloaded: int, duration: float):
        """Record a completed download for bandwidth estimation."""
        with self.lock:
            self.downloaded_bytes += bytes_downloaded
            self.completed_files += 1
            if duration > 0:
                speed = bytes_downloaded / duration
                self.recent_speeds.append(speed)

    def add_failure(self):
        """Record a failed download."""
        with self.lock:
            self.failed_files += 1

    def add_skip(self, bytes_size: int):
        """Record a skipped file (already downloaded)."""
        with self.lock:
            self.skipped_files += 1
            self.downloaded_bytes += bytes_size

    @property
    def avg_speed(self) -> float:
        """Calculate average download speed in bytes/second."""
        with self.lock:
            if not self.recent_speeds:
                elapsed = time.time() - self.start_time
                return self.downloaded_bytes / elapsed if elapsed > 0 else 0
            return sum(self.recent_speeds) / len(self.recent_speeds)

    @property
    def eta_seconds(self) -> float:
        """Estimate time remaining in seconds."""
        remaining_bytes = self.total_bytes - self.downloaded_bytes
        if self.avg_speed > 0:
            return remaining_bytes / self.avg_speed
        return 0

    def format_speed(self) -> str:
        """Format current speed as human-readable string."""
        speed = self.avg_speed
        if speed >= 1024 * 1024:
            return f"{speed / (1024 * 1024):.1f} MB/s"
        elif speed >= 1024:
            return f"{speed / 1024:.1f} KB/s"
        return f"{speed:.0f} B/s"

    def format_eta(self) -> str:
        """Format ETA as human-readable string."""
        eta = self.eta_seconds
        if eta <= 0:
            return "calculating..."
        if eta >= 3600:
            return f"{eta / 3600:.1f}h"
        elif eta >= 60:
            return f"{eta / 60:.0f}m"
        return f"{eta:.0f}s"


@dataclass
class DownloadState:
    """Persistent state for tracking download progress across restarts."""
    years_completed: List[int] = field(default_factory=list)
    current_year: Optional[int] = None
    total_downloaded_bytes: int = 0
    last_updated: str = ""

    def save(self, path: Path):
        """Save state to file."""
        self.last_updated = datetime.now().isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'DownloadState':
        """Load state from file, or return new state if file doesn't exist."""
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                    return cls(
                        years_completed=data.get('years_completed', []),
                        current_year=data.get('current_year'),
                        total_downloaded_bytes=data.get('total_downloaded_bytes', 0),
                        last_updated=data.get('last_updated', ''),
                    )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load state file: {e}")
        return cls()

    def mark_year_complete(self, year: int):
        """Mark a year as fully completed."""
        if year not in self.years_completed:
            self.years_completed.append(year)
            self.years_completed.sort()
        self.current_year = None

    def unmark_year_complete(self, year: int):
        """Remove a year from completed list (for re-checking)."""
        if year in self.years_completed:
            self.years_completed.remove(year)


class ConcurrencyController:
    """
    Controls concurrent downloads using a semaphore.

    Tracks active downloads and allows adjusting max concurrency dynamically.
    """

    def __init__(self, max_concurrent: int = MAX_CONCURRENT):
        self.max_concurrent = max_concurrent
        self._semaphore = Semaphore(max_concurrent)
        self._active = 0
        self._lock = Lock()

    def acquire(self):
        """Acquire a download slot (blocks if at max concurrency)."""
        self._semaphore.acquire()
        with self._lock:
            self._active += 1

    def release(self):
        """Release a download slot."""
        with self._lock:
            self._active -= 1
        self._semaphore.release()

    @property
    def active(self) -> int:
        """Get number of currently active downloads."""
        with self._lock:
            return self._active

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


def list_s3_objects(prefix: str, delimiter: str = "") -> Generator[dict, None, None]:
    """
    List objects from NOAA's S3-style object store with pagination support.

    Args:
        prefix: The prefix to filter objects (e.g., 'hourly/access/by-year/2024/parquet/')
        delimiter: Optional delimiter for folder-like listing

    Yields:
        Dictionary with object metadata (Key, Size, LastModified) or CommonPrefixes
    """
    continuation_token = None

    while True:
        params = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": "1000",
        }
        if delimiter:
            params["delimiter"] = delimiter
        if continuation_token:
            params["continuation-token"] = continuation_token

        retry_count = 0
        while retry_count < MAX_DOWNLOAD_RETRIES:
            try:
                response = requests.get(
                    GHCNH_BASE_URL,
                    params=params,
                    timeout=DOWNLOAD_TIMEOUT
                )
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count >= MAX_DOWNLOAD_RETRIES:
                    logger.error(f"Failed to list objects after {MAX_DOWNLOAD_RETRIES} attempts: {e}")
                    return
                wait_time = 2 ** retry_count
                logger.warning(f"Error listing objects: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        # Parse XML response
        root = ET.fromstring(response.content)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

        # Yield CommonPrefixes if using delimiter (for folder listing)
        for cp in root.findall(".//s3:CommonPrefixes", ns):
            prefix_elem = cp.find("s3:Prefix", ns)
            if prefix_elem is not None:
                yield {"Prefix": prefix_elem.text}

        # Yield Contents (actual objects)
        for contents in root.findall(".//s3:Contents", ns):
            key = contents.find("s3:Key", ns)
            size = contents.find("s3:Size", ns)
            last_modified = contents.find("s3:LastModified", ns)

            if key is not None:
                yield {
                    "Key": key.text,
                    "Size": int(size.text) if size is not None else 0,
                    "LastModified": last_modified.text if last_modified is not None else None,
                }

        # Check for pagination
        is_truncated = root.find(".//s3:IsTruncated", ns)
        if is_truncated is not None and is_truncated.text == "true":
            next_token = root.find(".//s3:NextContinuationToken", ns)
            if next_token is not None:
                continuation_token = next_token.text
            else:
                break
        else:
            break


def get_available_years() -> List[int]:
    """Get list of available years in the GHCNh dataset."""
    years = []
    prefix = "hourly/access/by-year/"

    logger.info("Fetching available years...")
    for obj in list_s3_objects(prefix, delimiter="/"):
        if "Prefix" in obj:
            # Extract year from prefix like 'hourly/access/by-year/2024/'
            match = re.search(r"/(\d{4})/$", obj["Prefix"])
            if match:
                years.append(int(match.group(1)))

    years.sort()
    logger.info(f"Found {len(years)} years: {years[0]} to {years[-1]}")
    return years


def get_parquet_files_for_year(year: int, us_only: bool = True) -> List[dict]:
    """
    Get list of parquet files for a specific year.

    Args:
        year: The year to get files for
        us_only: If True, only return US station files

    Returns:
        List of dicts with file metadata (Key, Size, LastModified, station_id)
    """
    prefix = f"hourly/access/by-year/{year}/parquet/"
    files = []

    logger.info(f"Listing parquet files for {year}...")
    for obj in list_s3_objects(prefix):
        if "Key" not in obj or not obj["Key"].endswith(".parquet"):
            continue

        # Extract station ID from filename like 'GHCNh_USW00003046_2024.parquet'
        filename = obj["Key"].split("/")[-1]
        match = re.match(r"GHCNh_([^_]+)_\d{4}\.parquet", filename)
        if not match:
            continue

        station_id = match.group(1)

        # Filter to US stations if requested
        if us_only and not station_id.startswith(NORTH_AMERICA_PREFIXES):
            continue

        obj["station_id"] = station_id
        obj["filename"] = filename
        files.append(obj)

    logger.info(f"Found {len(files)} {'US ' if us_only else ''}parquet files for {year}")
    return files


def download_file(
    url: str,
    dest_path: Path,
    file_size: int,
    stats: DownloadStats,
    concurrency: ConcurrencyController,
) -> bool:
    """
    Download a single file with retry mechanism.

    Args:
        url: URL to download from
        dest_path: Local path to save the file
        file_size: Expected file size for verification
        stats: DownloadStats object for tracking progress
        concurrency: ConcurrencyController for tracking active downloads

    Returns:
        True if download successful, False otherwise
    """
    retry_count = 0
    backoff_base = 2
    # Disable keep-alive to prevent stale connection issues
    headers = {"Connection": "close"}

    with concurrency:  # Acquire/release download slot automatically
        while retry_count < MAX_DOWNLOAD_RETRIES:
            start_time = time.time()
            try:
                # Check if file already exists with correct size
                if dest_path.exists():
                    existing_size = dest_path.stat().st_size
                    if file_size > 0 and existing_size == file_size:
                        stats.add_skip(file_size)
                        return True
                    # File exists but size mismatch, re-download
                    dest_path.unlink()

                with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, headers=headers) as response:
                    # Check for rate limiting
                    if response.status_code in RATE_LIMIT_CODES:
                        retry_count += 1
                        wait_time = backoff_base ** retry_count
                        logger.debug(f"Rate limited on {url}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue

                    response.raise_for_status()

                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    bytes_downloaded = 0

                    with open(dest_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                                bytes_downloaded += len(chunk)

                duration = time.time() - start_time
                stats.add_download(bytes_downloaded, duration)
                return True

            except requests.exceptions.HTTPError as e:
                retry_count += 1
                if dest_path.exists():
                    dest_path.unlink()

                if retry_count < MAX_DOWNLOAD_RETRIES:
                    wait_time = backoff_base ** retry_count
                    logger.debug(f"Error downloading {url}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to download {url} after {MAX_DOWNLOAD_RETRIES} attempts: {e}")
                    stats.add_failure()
                    return False

            except CONNECTION_RESET_ERRORS as e:
                # Connection was forcibly closed - common for long-running downloads
                retry_count += 1
                if dest_path.exists():
                    dest_path.unlink()

                if retry_count < MAX_DOWNLOAD_RETRIES:
                    # Use longer backoff for connection resets
                    wait_time = backoff_base ** (retry_count + 1)
                    logger.debug(f"Connection reset for {dest_path.name}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to download {url} after {MAX_DOWNLOAD_RETRIES} connection resets")
                    stats.add_failure()
                    return False

            except Exception as e:
                retry_count += 1
                if dest_path.exists():
                    dest_path.unlink()

                if retry_count < MAX_DOWNLOAD_RETRIES:
                    wait_time = backoff_base ** retry_count
                    logger.debug(f"Error downloading {url}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to download {url} after {MAX_DOWNLOAD_RETRIES} attempts: {e}")
                    stats.add_failure()
                    return False

    stats.add_failure()
    return False


def format_size(bytes_val: int) -> str:
    """Format bytes as human-readable size."""
    if bytes_val >= 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024 * 1024):.1f} GB"
    elif bytes_val >= 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.1f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.1f} KB"
    return f"{bytes_val} B"


def download_year(
    year: int,
    us_only: bool = True,
    force: bool = False,
    max_workers: int = MAX_DOWNLOAD_THREADS,
    state: Optional[DownloadState] = None,
    concurrency: Optional[ConcurrencyController] = None,
) -> Tuple[int, int]:
    """
    Download all parquet files for a specific year.

    Args:
        year: Year to download
        us_only: Only download US stations
        force: Re-download even if file exists
        max_workers: Maximum number of parallel downloads
        state: Optional download state for persistence
        concurrency: Optional concurrency controller

    Returns:
        Tuple of (successful_downloads, failed_downloads)
    """
    files = get_parquet_files_for_year(year, us_only=us_only)
    if not files:
        logger.warning(f"No files found for year {year}")
        return 0, 0

    year_dir = GHCNH_RAW_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    # Initialize concurrency controller if not provided
    if concurrency is None:
        concurrency = ConcurrencyController(max_concurrent=max_workers)

    # Prepare download tasks and calculate totals
    tasks = []
    stats = DownloadStats()
    stats.total_files = len(files)

    for file_info in files:
        dest_path = year_dir / file_info["filename"]
        stats.total_bytes += file_info["Size"]

        # Skip if file exists with correct size (unless force)
        if not force and dest_path.exists():
            if file_info["Size"] > 0 and dest_path.stat().st_size == file_info["Size"]:
                stats.add_skip(file_info["Size"])
                continue

        url = GHCNH_BASE_URL + file_info["Key"]
        tasks.append((url, dest_path, file_info["Size"], file_info["filename"]))

    if stats.skipped_files > 0:
        logger.info(f"Skipping {stats.skipped_files} already downloaded files for {year}")

    if not tasks:
        logger.info(f"All files for {year} already downloaded")
        if state:
            state.mark_year_complete(year)
            state.save(STATE_FILE)
        return stats.skipped_files, 0

    total_size = sum(t[2] for t in tasks)
    logger.info(f"Downloading {len(tasks)} files ({format_size(total_size)}) for {year}...")

    # Update state
    if state:
        state.current_year = year
        state.save(STATE_FILE)

    # Track failed files for retry
    failed_tasks = []
    failed_lock = Lock()

    def do_download_pass(task_list: List[tuple], pass_name: str = ""):
        """Execute a download pass and return failed tasks."""
        nonlocal stats
        pass_failed = []

        desc = f"Year {year}" if not pass_name else f"Year {year} ({pass_name})"

        with tqdm(
            total=len(task_list),
            desc=desc,
            unit="files",
            ncols=TQDM_NCOLS,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}] {postfix}",
        ) as pbar:

            def update_postfix():
                pbar.set_postfix_str(
                    f"{concurrency.active}/{max_workers} | {stats.format_speed()} | F:{stats.failed_files}"
                )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(download_file, url, path, size, stats, concurrency): (url, path, size, filename)
                    for url, path, size, filename in task_list
                }

                for future in as_completed(future_to_task):
                    url, path, size, filename = future_to_task[future]

                    try:
                        success = future.result()
                        if not success:
                            with failed_lock:
                                pass_failed.append((url, path, size, filename))
                    except Exception as e:
                        stats.add_failure()
                        logger.error(f"Exception downloading {filename}: {e}")
                        with failed_lock:
                            pass_failed.append((url, path, size, filename))

                    pbar.update(1)
                    update_postfix()

        return pass_failed

    # First pass
    failed_tasks = do_download_pass(tasks)

    # Retry pass for failed files
    if failed_tasks:
        logger.info(f"Retrying {len(failed_tasks)} failed files for {year}...")
        # Reset failure count for retry tracking
        stats.failed_files = 0

        # Give server a brief break before retries
        time.sleep(2)

        still_failed = do_download_pass(failed_tasks, "retry")

        # Update final failure count
        stats.failed_files = len(still_failed)

        if still_failed:
            logger.warning(f"{len(still_failed)} files still failed after retry:")
            for _, _, _, filename in still_failed[:10]:
                logger.warning(f"  - {filename}")
            if len(still_failed) > 10:
                logger.warning(f"  ... and {len(still_failed) - 10} more")

    # Final state save - only mark year complete if all files downloaded
    successful = stats.completed_files + stats.skipped_files
    if state:
        if stats.failed_files == 0:
            state.mark_year_complete(year)
        state.total_downloaded_bytes += stats.downloaded_bytes
        state.save(STATE_FILE)

    logger.info(
        f"Year {year}: {successful} successful, {stats.failed_files} failed "
        f"({format_size(stats.downloaded_bytes)} downloaded)"
    )

    return successful, stats.failed_files


def parse_year_filter(year_filter: str) -> List[int]:
    """
    Parse a year filter string into a list of years.

    Supports:
        - Single year: "2024"
        - Range: "2020:2024"
        - Comma-separated: "2020,2022,2024"
        - Mixed: "2010:2015,2020,2024"
    """
    years = set()

    for part in year_filter.split(","):
        part = part.strip()
        if ":" in part:
            start, end = part.split(":")
            years.update(range(int(start), int(end) + 1))
        else:
            years.add(int(part))

    return sorted(years)


def show_status():
    """Show current download status from saved state."""
    state = DownloadState.load(STATE_FILE)

    print("\n" + "=" * 60)
    print("GHCNh Download Status")
    print("=" * 60)

    if not state.last_updated:
        print("\nNo download state found. Run a download to create state.")
        return

    print(f"\nLast updated: {state.last_updated}")
    print(f"Total downloaded: {format_size(state.total_downloaded_bytes)}")
    print(f"Years completed: {len(state.years_completed)}")

    if state.years_completed:
        print(f"  Completed: {', '.join(str(y) for y in sorted(state.years_completed)[-10:])}")
        if len(state.years_completed) > 10:
            print(f"  ... and {len(state.years_completed) - 10} more")

    if state.current_year:
        # Count files on disk for current year
        year_dir = GHCNH_RAW_DIR / str(state.current_year)
        files_done = len(list(year_dir.glob("*.parquet"))) if year_dir.exists() else 0
        print(f"\nIn progress: {state.current_year} ({files_done} files on disk)")

    print("\nTo resume: python ghcnh_downloader.py")
    print("To start fresh: python ghcnh_downloader.py --force")
    print("To audit/verify: python ghcnh_downloader.py --audit")
    print("=" * 60 + "\n")


def show_inventory():
    """Download and display station inventory from NOAA."""
    import csv
    from io import StringIO
    from collections import Counter

    inventory_url = GHCNH_BASE_URL + "hourly/doc/ghcnh-station-list.csv"

    print("\n" + "=" * 60)
    print("GHCNh Station Inventory")
    print("=" * 60)
    print(f"\nFetching station list from NOAA...")

    try:
        response = requests.get(inventory_url, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch station inventory: {e}")
        print(f"\nError: Could not download station list: {e}")
        return

    # Parse CSV
    content = response.text
    reader = csv.DictReader(StringIO(content))

    # Count stations by country code (first 2 chars of station ID)
    country_counts = Counter()
    total_stations = 0

    for row in reader:
        # Try multiple possible column names for station ID
        station_id = row.get("GHCN_ID", row.get("station_id", row.get("id", "")))
        if len(station_id) >= 2:
            country_code = station_id[:2]
            country_counts[country_code] += 1
            total_stations += 1

    # Calculate North America counts
    na_count = sum(country_counts.get(code, 0) for code in NORTH_AMERICA_PREFIXES)

    print(f"\nTotal stations worldwide: {total_stations:,}")
    print(f"North America filter ({', '.join(NORTH_AMERICA_PREFIXES)}): {na_count:,}")

    # Show North America breakdown
    print("\n--- North America Breakdown ---")
    na_codes = [(code, country_counts.get(code, 0)) for code in NORTH_AMERICA_PREFIXES]
    na_codes.sort(key=lambda x: -x[1])
    for code, count in na_codes:
        if count > 0:
            print(f"  {code}: {count:,} stations")

    # Show top 10 other countries
    print("\n--- Top 10 Other Countries ---")
    other_codes = [(code, count) for code, count in country_counts.items()
                   if code not in NORTH_AMERICA_PREFIXES]
    other_codes.sort(key=lambda x: -x[1])
    for code, count in other_codes[:10]:
        print(f"  {code}: {count:,} stations")

    print("\n" + "=" * 60 + "\n")


def audit_downloads(years: List[int], us_only: bool = True, fix: bool = False, max_workers: int = MAX_DOWNLOAD_THREADS) -> dict:
    """
    Audit downloaded files against the server to find missing files.

    Args:
        years: List of years to audit
        us_only: Only check US stations
        fix: If True, download missing files

    Returns:
        Dictionary with audit results
    """
    results = {
        "years_checked": 0,
        "total_expected": 0,
        "total_found": 0,
        "total_missing": 0,
        "total_size_mismatch": 0,
        "missing_files": [],
        "size_mismatch_files": [],
        "fixed_success": 0,
        "fixed_failed": 0,
    }

    concurrency = ConcurrencyController(max_concurrent=max_workers) if fix else None

    for year in years:
        results["years_checked"] += 1
        year_dir = GHCNH_RAW_DIR / str(year)

        # Get expected files from server
        expected_files = get_parquet_files_for_year(year, us_only=us_only)
        results["total_expected"] += len(expected_files)

        missing = []
        size_mismatch = []

        for file_info in expected_files:
            local_path = year_dir / file_info["filename"]

            if not local_path.exists():
                missing.append(file_info)
            elif file_info["Size"] > 0 and local_path.stat().st_size != file_info["Size"]:
                size_mismatch.append(file_info)
            else:
                results["total_found"] += 1

        results["total_missing"] += len(missing)
        results["total_size_mismatch"] += len(size_mismatch)

        if missing:
            results["missing_files"].extend([(year, f["filename"]) for f in missing])
        if size_mismatch:
            results["size_mismatch_files"].extend([(year, f["filename"]) for f in size_mismatch])

        # Report for this year
        if missing or size_mismatch:
            logger.warning(f"Year {year}: {len(missing)} missing, {len(size_mismatch)} size mismatch")

            if fix and (missing or size_mismatch):
                # Download missing/mismatched files
                to_fix = missing + size_mismatch
                logger.info(f"Fixing {len(to_fix)} files for year {year}...")

                stats = DownloadStats()
                stats.total_files = len(to_fix)
                stats.total_bytes = sum(f["Size"] for f in to_fix)

                year_dir.mkdir(parents=True, exist_ok=True)

                tasks = [
                    (GHCNH_BASE_URL + f["Key"], year_dir / f["filename"], f["Size"], f["filename"])
                    for f in to_fix
                ]

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {
                        executor.submit(download_file, url, path, size, stats, concurrency): filename
                        for url, path, size, filename in tasks
                    }

                    for future in as_completed(future_to_file):
                        filename = future_to_file[future]
                        try:
                            success = future.result()
                            if success:
                                results["fixed_success"] += 1
                            else:
                                results["fixed_failed"] += 1
                        except Exception as e:
                            logger.error(f"Failed to fix {filename}: {e}")
                            results["fixed_failed"] += 1

                logger.info(f"Year {year}: fixed {stats.completed_files}/{len(to_fix)} files")
        else:
            logger.info(f"Year {year}: OK ({len(expected_files)} files)")

    return results


def clean_downloads(years: List[int], us_only: bool = True, dry_run: bool = True) -> dict:
    """
    Clean downloaded files by removing corrupted files and files not matching the station filter.

    Args:
        years: List of years to clean
        us_only: If True, remove files not matching NORTH_AMERICA_PREFIXES
        dry_run: If True, only report what would be deleted without actually deleting

    Returns:
        Dictionary with clean results
    """
    results = {
        "years_checked": 0,
        "files_checked": 0,
        "corrupted_files": [],
        "filtered_files": [],
        "deleted_count": 0,
        "deleted_bytes": 0,
    }

    for year in years:
        results["years_checked"] += 1
        year_dir = GHCNH_RAW_DIR / str(year)

        if not year_dir.exists():
            continue

        # Get expected files from server for size verification
        logger.info(f"Checking year {year}...")
        expected_files = get_parquet_files_for_year(year, us_only=False)  # Get all for size check
        expected_by_name = {f["filename"]: f for f in expected_files}

        # Also get the filtered set to know which stations are valid
        if us_only:
            valid_files = get_parquet_files_for_year(year, us_only=True)
            valid_names = {f["filename"] for f in valid_files}
        else:
            valid_names = set(expected_by_name.keys())

        # Check each local file
        local_files = list(year_dir.glob("*.parquet"))
        for local_path in local_files:
            results["files_checked"] += 1
            filename = local_path.name
            file_size = local_path.stat().st_size

            # Check if file matches station filter
            if filename not in valid_names:
                # Extract station ID to check if it's a filter issue
                match = re.match(r"GHCNh_([^_]+)_\d{4}\.parquet", filename)
                if match:
                    station_id = match.group(1)
                    if us_only and not station_id.startswith(NORTH_AMERICA_PREFIXES):
                        results["filtered_files"].append((year, filename, file_size))
                        if not dry_run:
                            local_path.unlink()
                            results["deleted_count"] += 1
                            results["deleted_bytes"] += file_size
                        continue

            # Check for size mismatch (corruption)
            if filename in expected_by_name:
                expected_size = expected_by_name[filename]["Size"]
                if expected_size > 0 and file_size != expected_size:
                    results["corrupted_files"].append((year, filename, file_size, expected_size))
                    if not dry_run:
                        local_path.unlink()
                        results["deleted_count"] += 1
                        results["deleted_bytes"] += file_size

        # Report for this year
        year_corrupted = len([f for f in results["corrupted_files"] if f[0] == year])
        year_filtered = len([f for f in results["filtered_files"] if f[0] == year])
        if year_corrupted or year_filtered:
            logger.info(f"Year {year}: {year_corrupted} corrupted, {year_filtered} filtered out")
        else:
            logger.info(f"Year {year}: OK ({len(local_files)} files)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download GHCNh parquet files from NOAA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ghcnh_downloader.py                     # Download all years, US stations only
    python ghcnh_downloader.py --year 2024         # Download specific year
    python ghcnh_downloader.py --year 2020:2024    # Download year range
    python ghcnh_downloader.py --all-stations      # Download all stations globally
    python ghcnh_downloader.py --list-years        # List available years
    python ghcnh_downloader.py --status            # Show download progress
    python ghcnh_downloader.py --audit             # Check for missing/corrupted files
    python ghcnh_downloader.py --audit --fix       # Download missing/corrupted files
    python ghcnh_downloader.py --clean             # Show files to delete (dry run)
    python ghcnh_downloader.py --clean --delete    # Delete corrupted/non-matching files
        """
    )
    parser.add_argument(
        "--year", "-y",
        type=str,
        help="Year(s) to download. Supports single year, ranges (2020:2024), or comma-separated (2020,2022)"
    )
    parser.add_argument(
        "--all-stations",
        action="store_true",
        help="Download all stations globally, not just US stations"
    )
    parser.add_argument(
        "--list-years",
        action="store_true",
        help="List available years and exit"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show download status and exit"
    )
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Download and display station inventory from NOAA (shows station counts by country)"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Audit downloaded files against server to find missing/corrupted files"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Used with --audit: automatically download missing/corrupted files"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove corrupted files and files not matching station filter (dry-run by default)"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Used with --clean: actually delete files (otherwise just shows what would be deleted)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode - only show warnings and errors"
    )
    parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=MAX_DOWNLOAD_THREADS,
        help=f"Maximum parallel downloads (default: {MAX_DOWNLOAD_THREADS})"
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Don't save/load download state (no restart capability)"
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    # Status mode
    if args.status:
        show_status()
        return

    # Inventory mode
    if args.inventory:
        show_inventory()
        return

    # Use custom max workers if specified
    max_workers = args.max_workers if args.max_workers else MAX_DOWNLOAD_THREADS

    # List years mode
    if args.list_years:
        years = get_available_years()
        print(f"\nAvailable years: {years[0]} to {years[-1]} ({len(years)} years)")
        print(f"\nTo download a specific year: python ghcnh_downloader.py --year {years[-1]}")
        print(f"To download a range: python ghcnh_downloader.py --year {years[-5]}:{years[-1]}")
        return

    # Determine which years to check/download
    available_years = get_available_years()

    if args.year:
        years_to_download = parse_year_filter(args.year)
        # Validate years
        invalid = [y for y in years_to_download if y not in available_years]
        if invalid:
            logger.error(f"Invalid years: {invalid}. Available: {available_years[0]}-{available_years[-1]}")
            return
    else:
        years_to_download = available_years

    us_only = not args.all_stations
    station_type = "US" if us_only else "all"

    # Audit mode
    if args.audit:
        logger.info(f"Auditing {len(years_to_download)} years ({station_type} stations)...")
        results = audit_downloads(
            years_to_download,
            us_only=us_only,
            fix=args.fix,
            max_workers=max_workers
        )

        print("\n" + "=" * 60)
        print("Audit Results")
        print("=" * 60)
        print(f"Years checked: {results['years_checked']}")
        print(f"Expected files: {results['total_expected']}")
        print(f"Found: {results['total_found']}")
        print(f"Missing: {results['total_missing']}")
        print(f"Size mismatch: {results['total_size_mismatch']}")

        if results['missing_files'] and not args.fix:
            print(f"\nMissing files (first 20):")
            for year, filename in results['missing_files'][:20]:
                print(f"  {year}/{filename}")
            if len(results['missing_files']) > 20:
                print(f"  ... and {len(results['missing_files']) - 20} more")
            print(f"\nTo fix: python ghcnh_downloader.py --audit --fix")

        if args.fix and (results['total_missing'] > 0 or results['total_size_mismatch'] > 0):
            total_to_fix = results['total_missing'] + results['total_size_mismatch']
            print(f"\nFix results:")
            print(f"  Attempted: {total_to_fix}")
            print(f"  Successful: {results['fixed_success']}")
            print(f"  Failed: {results['fixed_failed']}")
            if results['fixed_failed'] > 0:
                print(f"\nSome files failed to download. Run --audit --fix again to retry.")

        print("=" * 60 + "\n")
        return

    # Clean mode
    if args.clean:
        dry_run = not args.delete
        mode_str = "DRY RUN" if dry_run else "DELETING"
        logger.info(f"Cleaning {len(years_to_download)} years ({station_type} stations) - {mode_str}...")

        results = clean_downloads(
            years_to_download,
            us_only=us_only,
            dry_run=dry_run
        )

        print("\n" + "=" * 60)
        print(f"Clean Results {'(DRY RUN)' if dry_run else ''}")
        print("=" * 60)
        print(f"Years checked: {results['years_checked']}")
        print(f"Files checked: {results['files_checked']}")
        print(f"Corrupted files: {len(results['corrupted_files'])}")
        print(f"Non-matching stations: {len(results['filtered_files'])}")

        if results['corrupted_files']:
            print(f"\nCorrupted files (size mismatch):")
            for year, filename, actual, expected in results['corrupted_files'][:10]:
                print(f"  {year}/{filename}: {format_size(actual)} (expected {format_size(expected)})")
            if len(results['corrupted_files']) > 10:
                print(f"  ... and {len(results['corrupted_files']) - 10} more")

        if results['filtered_files']:
            print(f"\nNon-matching stations (not in {', '.join(NORTH_AMERICA_PREFIXES)}):")
            for year, filename, size in results['filtered_files'][:10]:
                print(f"  {year}/{filename} ({format_size(size)})")
            if len(results['filtered_files']) > 10:
                print(f"  ... and {len(results['filtered_files']) - 10} more")

        total_to_delete = len(results['corrupted_files']) + len(results['filtered_files'])
        total_bytes = sum(f[2] for f in results['filtered_files']) + sum(f[2] for f in results['corrupted_files'])

        if dry_run and total_to_delete > 0:
            print(f"\nWould delete {total_to_delete} files ({format_size(total_bytes)})")
            print(f"To actually delete: python ghcnh_downloader.py --clean --delete")
        elif not dry_run and results['deleted_count'] > 0:
            print(f"\nDeleted {results['deleted_count']} files ({format_size(results['deleted_bytes'])})")

        print("=" * 60 + "\n")
        return

    # Load or create state
    state = None if args.no_state else DownloadState.load(STATE_FILE)
    if state and not args.force:
        # Skip already completed years
        years_to_download = [y for y in years_to_download if y not in state.years_completed]
        if state.current_year and state.current_year in years_to_download:
            # Move current year to front
            years_to_download.remove(state.current_year)
            years_to_download.insert(0, state.current_year)

    if not years_to_download:
        logger.info("All requested years already downloaded. Use --force to re-download.")
        return

    logger.info(f"Starting download of {len(years_to_download)} years ({station_type} stations)")
    logger.info(f"Output directory: {GHCNH_RAW_DIR}")

    # Ensure output directory exists
    GHCNH_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Create shared concurrency controller
    concurrency = ConcurrencyController(max_concurrent=max_workers)

    # Download each year
    total_success = 0
    total_failed = 0
    start_time = time.time()

    for year in years_to_download:
        success, failed = download_year(
            year,
            us_only=us_only,
            force=args.force,
            max_workers=max_workers,
            state=state,
            concurrency=concurrency,
        )
        total_success += success
        total_failed += failed

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"Total files: {total_success + total_failed}")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_failed}")
    print(f"Time elapsed: {elapsed:.1f}s")
    if elapsed > 0:
        print(f"Average speed: {format_size(int((state.total_downloaded_bytes if state else 0) / elapsed))}/s")
    print(f"Output directory: {GHCNH_RAW_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
