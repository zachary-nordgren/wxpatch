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


if __name__ == "__main__":
    main()
