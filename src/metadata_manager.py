#!/usr/bin/env python3
"""
SQLite-based metadata manager for the NOAA weather data processor.
Provides atomic updates and efficient concurrent access.
"""
import sqlite3
import csv
import uuid
import logging
import threading
import shutil
import tarfile
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
from contextlib import contextmanager
from tqdm import tqdm

from config import MERGED_DIR, METADATA_FIELDS
from utils import safe_read_from_tar

logger = logging.getLogger("weather_processor")

# Database file path
DB_PATH = MERGED_DIR / "wx_metadata.db"
LEGACY_CSV_PATH = MERGED_DIR / "wx_info.csv"

# Thread-local storage for database connections
_thread_local = threading.local()

# Schema version
SCHEMA_VERSION = 1

# Lock for schema initialization
_init_lock = threading.Lock()
_db_semaphore = threading.Semaphore(3)  # Limit concurrent DB operations
_is_initialized = False


@contextmanager
def get_db_connection():
    """
    Get a database connection for the current thread.
    Uses thread-local storage to maintain separate connections per thread.
    Uses a semaphore to limit concurrent database operations.
    """
    global _is_initialized

    # If not initialized, initialize first
    if not _is_initialized:
        _initialize_database_internal()

    # Acquire semaphore with timeout to prevent deadlocks
    if not _db_semaphore.acquire(timeout=60):  # 60 second timeout
        logger.warning(
            "Failed to acquire database semaphore after 60 seconds, proceeding anyway"
        )

    try:
        # Check if we already have a connection for this thread
        if not hasattr(_thread_local, "connection"):
            # Create a new connection for this thread
            _thread_local.connection = sqlite3.connect(
                DB_PATH,
                timeout=30.0,  # Increased from 10 to 30 seconds
                isolation_level=None,  # Use autocommit mode
            )
            # Enable foreign keys
            _thread_local.connection.execute("PRAGMA foreign_keys = ON")

            # Set busy timeout - increased to 30 seconds
            _thread_local.connection.execute("PRAGMA busy_timeout = 30000")

            # Use write-ahead logging for better concurrency
            _thread_local.connection.execute("PRAGMA journal_mode = WAL")

        try:
            yield _thread_local.connection
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                # Implement retry logic with exponential backoff
                retry_count = 0
                max_retries = 5
                while retry_count < max_retries:
                    retry_count += 1
                    wait_time = (
                        2**retry_count
                    )  # Exponential backoff: 2, 4, 8, 16, 32 seconds
                    logger.warning(
                        f"Database locked, retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    try:
                        yield _thread_local.connection
                        return  # Success, exit the retry loop
                    except sqlite3.OperationalError as retry_e:
                        if (
                            retry_count == max_retries
                            or "database is locked" not in str(retry_e)
                        ):
                            logger.error(
                                f"Database error after retries: {retry_e}",
                                exc_info=True,
                            )
                            raise
            else:
                # Other database errors
                logger.error(f"Database error: {e}", exc_info=True)
                raise
    finally:
        # Always release the semaphore
        _db_semaphore.release()


def initialize_database():
    """
    Initialize the SQLite database with the required schema.
    Thread-safe and only runs once.
    """
    global _is_initialized

    # Check if we need to initialize
    if _is_initialized:
        return

    _initialize_database_internal()


def _initialize_database_internal():
    """
    Internal function to handle database initialization.
    This avoids the circular dependency between get_db_connection and initialize_database.
    """
    global _is_initialized

    with _init_lock:
        # Check again inside the lock
        if _is_initialized:
            return

        # Make sure the parent directory exists
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        create_new = not DB_PATH.exists()

        # Create a direct connection for initialization without get_db_connection
        conn = sqlite3.connect(DB_PATH, timeout=30.0, isolation_level=None)

        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 10000")
            conn.execute("PRAGMA journal_mode = WAL")

            cursor = conn.cursor()

            # Get current schema version if database exists
            if not create_new:
                try:
                    cursor.execute(
                        "SELECT value FROM metadata_settings WHERE key = 'schema_version'"
                    )
                    result = cursor.fetchone()
                    current_version = int(result[0]) if result else 0
                except sqlite3.OperationalError:
                    # Table doesn't exist yet - treat as new
                    current_version = 0
                    create_new = True
            else:
                current_version = 0

            # Create tables if needed
            if create_new or current_version < SCHEMA_VERSION:
                logger.info(f"Initializing metadata database (v{SCHEMA_VERSION})")

                # Settings table
                cursor.execute(
                    """
                CREATE TABLE IF NOT EXISTS metadata_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
                )

                # Stations table
                cursor.execute(
                    """
                CREATE TABLE IF NOT EXISTS stations (
                    station_id TEXT PRIMARY KEY,
                    latitude TEXT,
                    longitude TEXT,
                    elevation TEXT,
                    name TEXT,
                    call_sign TEXT,
                    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
                )

                # Year counts table with foreign key to stations
                cursor.execute(
                    """
                CREATE TABLE IF NOT EXISTS year_counts (
                    station_id TEXT,
                    year TEXT,
                    observation_count INTEGER,
                    PRIMARY KEY (station_id, year),
                    FOREIGN KEY (station_id) REFERENCES stations(station_id) ON DELETE CASCADE
                )
                """
                )

                # Create index on year for efficient filtering
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_year_counts_year ON year_counts(year)"
                )

                # Set schema version
                cursor.execute(
                    "INSERT OR REPLACE INTO metadata_settings (key, value) VALUES (?, ?)",
                    ("schema_version", str(SCHEMA_VERSION)),
                )

                # Import existing metadata if available
                if LEGACY_CSV_PATH.exists():
                    # Close this connection first to avoid locks
                    conn.close()
                    import_from_csv(LEGACY_CSV_PATH)
                    # Reconnect after import
                    conn = sqlite3.connect(DB_PATH, timeout=30.0, isolation_level=None)

                logger.info("Database initialization complete")

            _is_initialized = True

        finally:
            # Ensure connection is closed
            conn.close()


def import_from_csv(csv_path: Path):
    """
    Import metadata from the legacy CSV file.
    """
    logger.info(f"Importing metadata from {csv_path}")

    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            headers = next(reader, METADATA_FIELDS)

            # Identify which columns correspond to metadata fields vs year counts
            metadata_indices = {}
            year_indices = {}

            for i, header in enumerate(headers):
                if header in METADATA_FIELDS:
                    metadata_indices[header] = i
                elif header.isdigit():  # Year columns
                    year_indices[header] = i

            # Process rows using efficient batch inserts
            batch_size = 1000
            stations_batch = []
            year_counts_batch = []

            for row in reader:
                if not row or len(row) == 0 or not row[0]:
                    continue

                station_id = row[0]

                # Extract station metadata
                station_data = [
                    station_id,
                    (
                        row[metadata_indices.get("LATITUDE", 0)]
                        if "LATITUDE" in metadata_indices
                        else ""
                    ),
                    (
                        row[metadata_indices.get("LONGITUDE", 0)]
                        if "LONGITUDE" in metadata_indices
                        else ""
                    ),
                    (
                        row[metadata_indices.get("ELEVATION", 0)]
                        if "ELEVATION" in metadata_indices
                        else ""
                    ),
                    (
                        row[metadata_indices.get("NAME", 0)]
                        if "NAME" in metadata_indices
                        else ""
                    ),
                    (
                        row[metadata_indices.get("CALL_SIGN", 0)]
                        if "CALL_SIGN" in metadata_indices
                        else ""
                    ),
                ]

                stations_batch.append(station_data)

                # Extract year counts
                for year, idx in year_indices.items():
                    if idx < len(row) and row[idx]:
                        try:
                            count = int(row[idx])
                            year_counts_batch.append((station_id, year, count))
                        except ValueError:
                            pass

                # Execute batch insert if batch is full
                if len(stations_batch) >= batch_size:
                    with get_db_connection() as conn:
                        # Insert stations
                        conn.executemany(
                            """
                            INSERT OR REPLACE INTO stations 
                            (station_id, latitude, longitude, elevation, name, call_sign)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            stations_batch,
                        )

                        # Insert year counts
                        if year_counts_batch:
                            conn.executemany(
                                """
                                INSERT OR REPLACE INTO year_counts
                                (station_id, year, observation_count)
                                VALUES (?, ?, ?)
                                """,
                                year_counts_batch,
                            )

                    # Clear batches
                    stations_batch = []
                    year_counts_batch = []

            # Insert any remaining rows
            if stations_batch:
                with get_db_connection() as conn:
                    # Insert stations
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO stations 
                        (station_id, latitude, longitude, elevation, name, call_sign)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        stations_batch,
                    )

                    # Insert year counts
                    if year_counts_batch:
                        conn.executemany(
                            """
                            INSERT OR REPLACE INTO year_counts
                            (station_id, year, observation_count)
                            VALUES (?, ?, ?)
                            """,
                            year_counts_batch,
                        )

            logger.info(f"Imported metadata for {len(stations_batch)} stations")

    except Exception as e:
        logger.error(f"Error importing from CSV: {e}", exc_info=True)


def update_station_metadata(station_id: str, metadata: Dict[str, str]):
    """
    Update metadata for a single station.
    Efficiently handles concurrent updates with SQLite's ACID properties.

    Args:
        station_id: Station ID to update
        metadata: Dictionary of metadata values
    """
    try:
        # Prepare values
        values = (
            station_id,
            metadata.get("LATITUDE", ""),
            metadata.get("LONGITUDE", ""),
            metadata.get("ELEVATION", ""),
            metadata.get("NAME", ""),
            metadata.get("CALL_SIGN", ""),
        )

        # Use upsert pattern for atomic update
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO stations 
                (station_id, latitude, longitude, elevation, name, call_sign)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(station_id) DO UPDATE SET
                    latitude = CASE WHEN excluded.latitude != '' THEN excluded.latitude ELSE latitude END,
                    longitude = CASE WHEN excluded.longitude != '' THEN excluded.longitude ELSE longitude END,
                    elevation = CASE WHEN excluded.elevation != '' THEN excluded.elevation ELSE elevation END,
                    name = CASE WHEN excluded.name != '' THEN excluded.name ELSE name END,
                    call_sign = CASE WHEN excluded.call_sign != '' THEN excluded.call_sign ELSE call_sign END,
                    last_modified = CURRENT_TIMESTAMP
                """,
                values,
            )

    except Exception as e:
        logger.error(
            f"Error updating station {station_id} metadata: {e}", exc_info=True
        )


def update_station_year_count(station_id: str, year: str, count: int):
    """
    Update observation count for a station in a specific year.

    Args:
        station_id: Station ID
        year: Year as string (e.g., "2015")
        count: Number of observations
    """
    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO year_counts (station_id, year, observation_count)
                VALUES (?, ?, ?)
                ON CONFLICT(station_id, year) DO UPDATE SET
                    observation_count = excluded.observation_count
                """,
                (station_id, year, count),
            )
    except Exception as e:
        logger.error(
            f"Error updating year count for {station_id}, {year}: {e}", exc_info=True
        )


def update_batch_year_counts(counts_data: List[Tuple[str, str, int]]):
    """
    Update multiple year counts in a batch operation.
    Much more efficient than individual updates.

    Args:
        counts_data: List of (station_id, year, count) tuples
    """
    if not counts_data:
        return

    try:
        with get_db_connection() as conn:
            conn.execute("BEGIN TRANSACTION")

            # Use executemany for efficiency
            conn.executemany(
                """
                INSERT INTO year_counts (station_id, year, observation_count)
                VALUES (?, ?, ?)
                ON CONFLICT(station_id, year) DO UPDATE SET
                    observation_count = excluded.observation_count
                """,
                counts_data,
            )

            conn.execute("COMMIT")
    except Exception as e:
        logger.error(f"Error batch updating year counts: {e}", exc_info=True)


def get_all_station_metadata() -> Dict[str, Dict[str, str]]:
    """
    Get metadata for all stations.

    Returns:
        Dictionary mapping station_id to metadata dictionary
    """
    result = {}

    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT station_id, latitude, longitude, elevation, name, call_sign
                FROM stations
                """
            )

            for row in cursor:
                station_id, lat, lon, elev, name, call_sign = row
                result[station_id] = {
                    "STATION": station_id,
                    "LATITUDE": lat,
                    "LONGITUDE": lon,
                    "ELEVATION": elev,
                    "NAME": name,
                    "CALL_SIGN": call_sign,
                }
    except Exception as e:
        logger.error(f"Error getting station metadata: {e}", exc_info=True)

    return result


def count_observations_by_year(
    all_archives: List[Path], stations: Optional[Set[str]] = None
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
        # Extract just the year part from the filename (remove .tar.gz)
        year = archive.stem.split(".")[0]
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


def update_year_counts(
    all_archives: List[Path], updated_stations: Optional[Set[str]] = None
):
    """
    Update the year counts in the metadata database.

    Args:
        all_archives: List of paths to tar.gz archives
        updated_stations: Optional set of station IDs to update. If None, update all stations.
    """
    if updated_stations is not None and not updated_stations:
        logger.info("No stations updated, skipping year count update")
        return

    # Get observation counts
    logger.info("Counting observations by year...")
    observations, years = count_observations_by_year(all_archives, updated_stations)

    # Prepare data for batch update
    counts_data = []
    for station_id, year_counts in observations.items():
        for year, count in year_counts.items():
            counts_data.append((station_id, year, count))

    # Update in batches
    if counts_data:
        logger.info(
            f"Updating year counts for {len(observations)} stations across {len(years)} years"
        )

        # Process in batches of 5000 to avoid SQLite limits
        batch_size = 5000
        for i in range(0, len(counts_data), batch_size):
            batch = counts_data[i : i + batch_size]
            update_batch_year_counts(batch)

        logger.info("Year counts update completed")
    else:
        logger.info("No year counts to update")


def update_metadata_with_counts(observation_counts: Dict[str, Dict[str, int]]) -> bool:
    """
    Update metadata with observation counts.

    Args:
        observation_counts: Dictionary mapping station_id -> year -> count

    Returns:
        True if update succeeded, False otherwise
    """
    try:
        # Convert to list of tuples for batch update
        counts_data = []
        for station_id, year_counts in observation_counts.items():
            for year, count in year_counts.items():
                counts_data.append((station_id, year, count))

        # Update in batches
        if counts_data:
            logger.info(
                f"Updating observation counts for {len(observation_counts)} stations"
            )

            # Process in batches of 1000 to avoid SQLite limits (reduced from 5000)
            batch_size = 1000
            total_batches = (len(counts_data) + batch_size - 1) // batch_size

            for i in range(0, len(counts_data), batch_size):
                batch = counts_data[i : i + batch_size]
                batch_num = i // batch_size + 1

                logger.debug(
                    f"Processing batch {batch_num}/{total_batches} with {len(batch)} records"
                )
                try:
                    update_batch_year_counts(batch)
                    logger.debug(f"Completed batch {batch_num}/{total_batches}")
                except Exception as e:
                    logger.error(f"Error in batch {batch_num}: {e}", exc_info=True)

            logger.info("Observation counts update completed")
            return True
        else:
            logger.info("No observation counts to update")
            return True

    except Exception as e:
        logger.error(f"Error updating metadata with counts: {e}", exc_info=True)
        return False


def export_to_csv(output_path: Optional[Path] = None) -> bool:
    """
    Export metadata to CSV format.
    Generates a file compatible with the original format.

    Args:
        output_path: Optional path for output file. If None, uses the default wx_info.csv path.

    Returns:
        True if export succeeded, False otherwise
    """
    if output_path is None:
        output_path = LEGACY_CSV_PATH

    temp_path = output_path.with_name(f"wx_info_export_{uuid.uuid4()}.csv")

    try:
        logger.info(f"Exporting metadata to {output_path}")

        # Get years in ascending order
        with get_db_connection() as conn:
            cursor = conn.execute("SELECT DISTINCT year FROM year_counts ORDER BY year")
            years = [row[0] for row in cursor]

        # Define headers
        headers = METADATA_FIELDS + years

        # Open output file
        with open(temp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(headers)

            # Query all stations and their counts
            with get_db_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT s.station_id, s.latitude, s.longitude, s.elevation, s.name, s.call_sign
                    FROM stations s
                    ORDER BY s.station_id
                    """
                )

                batch_size = 1000
                stations_processed = 0

                while True:
                    stations_batch = cursor.fetchmany(batch_size)
                    if not stations_batch:
                        break

                    for station_row in stations_batch:
                        station_id = station_row[0]

                        # Create row with metadata
                        row = list(station_row)

                        # Get year counts for this station
                        year_cursor = conn.execute(
                            """
                            SELECT year, observation_count
                            FROM year_counts
                            WHERE station_id = ?
                            ORDER BY year
                            """,
                            (station_id,),
                        )

                        year_counts = dict(year_cursor.fetchall())

                        # Add year counts to row
                        for year in years:
                            row.append(str(year_counts.get(year, "")))

                        # Write row
                        writer.writerow(row)

                    stations_processed += len(stations_batch)
                    logger.debug(f"Exported {stations_processed} stations")

        # Rename to final path
        if output_path.exists():
            backup_path = output_path.with_suffix(".csv.bak")
            try:
                shutil.copy(output_path, backup_path)
                logger.debug(f"Created backup of existing file: {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")

            try:
                output_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove existing file: {e}")

        # Move temp file to final location
        temp_path.rename(output_path)

        logger.info(f"Exported metadata to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error exporting metadata to CSV: {e}", exc_info=True)
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e2:
                logger.warning(f"Failed to cleanup temp file: {e2}")
        return False


def recover_corrupted_metadata() -> bool:
    """
    Attempt to recover corrupted metadata.
    Since we're using SQLite, this just ensures the DB is properly initialized.

    Returns:
        True if recovery was successful
    """
    try:
        # Re-initialize database - this will import from CSV if available
        global _is_initialized
        _is_initialized = False
        initialize_database()

        # Perform integrity check
        with get_db_connection() as conn:
            cursor = conn.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]

            if result != "ok":
                logger.error(f"Database integrity check failed: {result}")

                # Try to recover by exporting to CSV, deleting DB and reimporting
                if export_to_csv():
                    # Delete database files
                    DB_PATH.unlink(missing_ok=True)
                    for related_file in DB_PATH.parent.glob(f"{DB_PATH.name}-*"):
                        related_file.unlink(missing_ok=True)

                    # Reinitialize
                    _is_initialized = False
                    initialize_database()

                    return True
            else:
                return True

    except Exception as e:
        logger.error(f"Error recovering metadata: {e}", exc_info=True)

    return False


def close_all_connections():
    """
    Close all database connections.
    Call this before program exit.
    """
    if hasattr(_thread_local, "connection"):
        try:
            _thread_local.connection.close()
            delattr(_thread_local, "connection")
        except Exception as e:
            logger.warning(f"Error closing database connection: {e}")


def finalize_metadata():
    """
    Perform final tasks before program exit.
    - Exports to CSV format
    - Closes database connections

    Add this to the end of merge_station_data function in processor_core.py.
    """
    try:
        # Export to CSV for compatibility with other tools
        export_to_csv()

        # Close connections
        close_all_connections()

        logger.info("Metadata finalization complete")

    except Exception as e:
        logger.error(f"Error finalizing metadata: {e}", exc_info=True)
