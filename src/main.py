#!/usr/bin/env python3
"""
Entry point for the NOAA weather data downloader and processor.
This script orchestrates the downloading and processing of NOAA global hourly weather data.
"""
import logging
import argparse
import time

# Import local modules
from config import (
    RAW_DIR,
    LOG_FILE,
    DEFAULT_MAX_WORKERS,
    DOWNLOAD_ONLY_OPTION,
    PROCESS_ONLY_OPTION,
    MAX_WORKERS_OPTION,
    YEAR_FILTER_OPTION,
    UPDATE_COUNTS_OPTION,
    RECOVER_METADATA_OPTION,
    VERBOSE_OPTION,
    QUIET_OPTION,
    NEWEST_FIRST_OPTION,
    SORT_CHRONOLOGICALLY_OPTION,
    MERGED_DIR,
    DATA_DIR,
)
from downloader import get_remote_file_list, download_archives
from file_io import setup_directories
from processor_core import merge_station_data, sort_station_files_chronologically
from metadata_manager import update_year_counts, recover_corrupted_metadata
from utils import parse_year_filter

# Set up logging with process ID for clarity between threads - log to file only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [PID:%(process)d] - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE)],
)
logger = logging.getLogger("weather_processor")


def main():
    """Main function to orchestrate the entire process."""
    parser = argparse.ArgumentParser(
        description="Process NOAA Global Hourly weather data"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean up log files, merged data, and temp directories (preserves raw downloads)",
    )
    parser.add_argument(
        DOWNLOAD_ONLY_OPTION,
        action="store_true",
        help="Only download archives without processing",
    )
    parser.add_argument(
        PROCESS_ONLY_OPTION,
        action="store_true",
        help="Only process existing archives without downloading",
    )
    parser.add_argument(
        MAX_WORKERS_OPTION,
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum number of concurrent archive extractions (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        YEAR_FILTER_OPTION,
        type=str,
        help="Process only archives for specific years (comma-separated)",
    )
    parser.add_argument(
        UPDATE_COUNTS_OPTION,
        action="store_true",
        help="Update observation counts for all stations",
    )
    parser.add_argument(
        RECOVER_METADATA_OPTION,
        action="store_true",
        help="Attempt to recover corrupted metadata file",
    )
    parser.add_argument(
        VERBOSE_OPTION, action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(QUIET_OPTION, action="store_true", help="Reduce logging output")
    parser.add_argument(
        NEWEST_FIRST_OPTION,
        action="store_true",
        help="Process newest years first (default: oldest first)",
    )
    parser.add_argument(
        SORT_CHRONOLOGICALLY_OPTION,
        action="store_true",
        help="Sort station data chronologically after processing",
    )
    args = parser.parse_args()

    # Run cleanup if requested and exit
    if args.clean:
        clean_directories()
        return

    # Configure logging level based on verbosity
    if args.verbose:
        logging.getLogger("weather_processor").setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger("weather_processor").setLevel(logging.WARNING)

    start_time = time.time()
    logger.info("Starting weather data processing")

    # Setup directories
    setup_directories()

    # Attempt metadata recovery if requested
    if args.recover_metadata:
        logger.info("Attempting to recover corrupted metadata file")
        if recover_corrupted_metadata():
            logger.info("Metadata recovery completed successfully")
        else:
            logger.warning("Metadata recovery failed or was not needed")

    # Download or update archives
    if not args.process_only:
        remote_files = get_remote_file_list()
        _ = download_archives(remote_files)

    # Get list of all archives
    all_archives = list(RAW_DIR.glob("*.tar.gz"))

    # Apply year filter if specified
    if args.year_filter:
        filter_years = parse_year_filter(args.year_filter)
        logger.info(f"Filtering to years: {', '.join(filter_years)}")
        all_archives = [a for a in all_archives if a.name.split(".")[0] in filter_years]

    # Sort archives - process in reverse chronological order if requested
    all_archives.sort(reverse=args.newest_first)

    logger.info(f"Found {len(all_archives)} archives to process")

    # Process archives
    updated_stations = set()
    if not args.download_only and all_archives:
        try:
            # Merge station data
            updated_stations = merge_station_data(
                all_archives, max_workers=args.max_workers
            )

            if args.update_counts:
                try:
                    logger.info(
                        "Running full observation count update for all stations..."
                    )
                    update_year_counts(all_archives, set())
                    logger.info("Year count update completed")
                except Exception as e:
                    logger.error(f"Error updating year counts: {e}")
                    logger.info("Attempting to recover metadata file...")
                    if recover_corrupted_metadata():
                        logger.info("Retrying year count update after recovery...")
                        try:
                            update_year_counts(all_archives, set())
                            logger.info("Year count update completed after recovery")
                        except Exception as retry_e:
                            logger.error(
                                f"Year count update failed even after recovery: {retry_e}"
                            )
        except Exception as e:
            logger.error(f"Error during data processing: {e}")

    # Sort station files chronologically if requested
    if args.sort_chronologically:
        logger.info("Sorting station files chronologically...")
        sort_station_files_chronologically()

    elapsed_time = time.time() - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Total stations updated: {len(updated_stations)}")


def clean_directories():
    """
    Clean up log files, merged data, and temp directories.
    Preserves raw data downloads.
    """
    import shutil
    import os
    import time
    from pathlib import Path

    print("Starting cleanup operation")

    # Clean log file
    log_path = Path(LOG_FILE)
    if log_path.exists():
        try:
            # Close logger handlers first to release file handle
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                logging.getLogger().removeHandler(handler)

            log_path.unlink()
            print(f"Deleted log file: {log_path}")
        except Exception as e:
            print(f"Error deleting log file: {e}")

    # Clean merged directory (but don't delete the directory itself)
    if MERGED_DIR.exists():
        try:
            # Delete the SQLite database if it exists
            db_path = MERGED_DIR / "wx_metadata.db"
            if db_path.exists():
                db_path.unlink()
                print(f"Deleted metadata database: {db_path}")

            # Also remove any WAL and SHM files
            for related_file in MERGED_DIR.glob("wx_metadata.db-*"):
                related_file.unlink()
                print(f"Deleted related database file: {related_file}")

            # Delete CSV metadata file if it exists
            metadata_path = MERGED_DIR / "wx_info.csv"
            if metadata_path.exists():
                metadata_path.unlink()
                print(f"Deleted metadata CSV: {metadata_path}")

            # Delete all gzipped station files
            count = 0
            for station_file in MERGED_DIR.glob("*.csv.gz"):
                station_file.unlink()
                count += 1

            print(f"Deleted {count} station files from {MERGED_DIR}")

            # Delete any backup files
            backup_count = 0
            for backup_file in MERGED_DIR.glob("*.bak*"):
                backup_file.unlink()
                backup_count += 1

            if backup_count > 0:
                print(f"Deleted {backup_count} backup files")

        except Exception as e:
            print(f"Error cleaning merged directory: {e}")

    # Clean temp directory and its contents
    temp_dir = DATA_DIR / "temp"
    if temp_dir.exists():
        try:
            # First try using rmtree with ignore_errors=True
            print(f"Attempting to delete temp directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

            # Check if directory still exists (might happen on Windows if files are locked)
            if temp_dir.exists():
                print("Temp directory still exists, trying more aggressive approach")

                # On Windows, try more aggressive approach
                if os.name == "nt":
                    # Try to forcefully close any handles to the directory
                    os.system(f"taskkill /F /IM python.exe /T")
                    time.sleep(1)  # Give OS time to release handles

                    # Try rmtree again
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass

                # If it still exists, try to clean contents
                if temp_dir.exists():
                    print("Could not remove directory, trying to delete contents")

                    # Try deleting the contents using a temporary directory
                    new_temp = temp_dir.with_name(f"temp_new_{int(time.time())}")
                    if not new_temp.exists():
                        new_temp.mkdir(parents=True, exist_ok=True)

                    # For future runs, we'll use the new directory for temp
                    with open(DATA_DIR / "temp_path.txt", "w") as f:
                        f.write(str(new_temp))

                    print(f"Created new temp directory at {new_temp}")
                    print(f"Please manually delete {temp_dir} when possible")
                else:
                    print(f"Successfully deleted temp directory after retry")
            else:
                print(f"Successfully deleted temp directory: {temp_dir}")

        except Exception as e:
            print(f"Error during temp directory cleanup: {e}")

            # Try to delete contents if full directory removal failed
            try:
                for item in temp_dir.glob("**/*"):
                    if item.is_file():
                        try:
                            item.unlink()
                            print(f"Deleted file: {item}")
                        except Exception:
                            print(f"Could not delete: {item}")
                    elif item.is_dir():
                        try:
                            shutil.rmtree(item, ignore_errors=True)
                            print(f"Deleted directory: {item}")
                        except Exception:
                            print(f"Could not delete: {item}")
                print(f"Deleted contents of temp directory: {temp_dir}")
            except Exception as nested_e:
                print(f"Error deleting temp directory contents: {nested_e}")

    print("Cleanup completed - raw downloads preserved")
    return True


if __name__ == "__main__":
    main()
