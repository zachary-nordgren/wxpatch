#!/usr/bin/env python3
"""
Functions for downloading NOAA weather data archives.
"""
import re
import time
import logging
import concurrent.futures
from datetime import datetime
from urllib.parse import urljoin
import requests
from tqdm import tqdm

# Import from config
from config import (
    BASE_URL,
    RAW_DIR,
    DOWNLOAD_TIMEOUT,
    DOWNLOAD_CHUNK_SIZE,
    MAX_DOWNLOAD_RETRIES,
    MAX_DOWNLOAD_THREADS,
)

logger = logging.getLogger("weather_processor")


def get_remote_file_list() -> list:
    """
    Get list of available tar.gz files from the NOAA website.
    Returns list of tuples: (filename, modified_date, file_size)
    """
    logger.info(f"Fetching file list from {BASE_URL}")

    retry_count = 0
    backoff_time = 2

    while retry_count < MAX_DOWNLOAD_RETRIES:
        try:
            response = requests.get(BASE_URL, timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()

            file_pattern = r'<tr><td><a href="([^"]+)">([^<]+)</a></td><td align="right">(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s+</td><td align="right">\s*([^<]+)</td>'
            matches = re.findall(file_pattern, response.text)

            file_list = []
            for _, filename, date_str, size in matches:
                if filename.endswith(".tar.gz"):
                    try:
                        modified_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
                        file_list.append((filename, modified_date, size))
                    except ValueError:
                        logger.warning(
                            f"Could not parse date for {filename}: {date_str}"
                        )

            logger.info(f"Found {len(file_list)} tar.gz files on the server")
            return file_list

        except requests.exceptions.RequestException as e:
            retry_count += 1
            wait_time = backoff_time**retry_count
            logger.warning(
                f"Error fetching file list: {e}. Retrying in {wait_time} seconds (attempt {retry_count}/{MAX_DOWNLOAD_RETRIES})"
            )
            time.sleep(wait_time)

    logger.error(f"Failed to fetch file list after {MAX_DOWNLOAD_RETRIES} attempts")
    return []


def download_file_with_retry(url, dest_path, year):
    """Download a file with retry mechanism for network resilience."""
    retry_count = 0

    while retry_count < MAX_DOWNLOAD_RETRIES:
        try:
            if dest_path.exists():
                logger.info(f"File already exists: {dest_path}")
                return True

            logger.info(f"Downloading {url} to {dest_path}")
            with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                # Use tqdm for progress tracking
                with open(dest_path, "wb") as f:
                    with tqdm(
                        desc=f"Downloading {year}",
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        leave=False,
                    ) as pbar:
                        for chunk in response.iter_content(
                            chunk_size=DOWNLOAD_CHUNK_SIZE
                        ):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

            logger.info(f"Download completed: {dest_path}")
            return True

        except Exception as e:
            retry_count += 1
            if dest_path.exists():
                dest_path.unlink()

            if retry_count < MAX_DOWNLOAD_RETRIES:
                wait_time = 2**retry_count  # Exponential backoff
                logger.warning(
                    f"Error downloading {url}: {e}. Retrying in {wait_time} seconds (attempt {retry_count}/{MAX_DOWNLOAD_RETRIES})"
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Failed to download {url} after {MAX_DOWNLOAD_RETRIES} attempts: {e}"
                )
                return False


def download_archives(file_list) -> list:
    """Download archives that are newer than local copies or don't exist locally."""
    downloads = []

    for filename, remote_date, size in file_list:
        local_path = RAW_DIR / filename
        year = filename.replace(".tar.gz", "")

        # Check if we need to download this file
        download_needed = not local_path.exists()

        if local_path.exists():
            local_date = datetime.fromtimestamp(local_path.stat().st_mtime)
            if remote_date > local_date:
                download_needed = True
                logger.info(f"Remote file {filename} ({size}) is newer than local copy")
            else:
                logger.info(f"Skipping {filename} - local copy is up to date")

        if download_needed:
            downloads.append((filename, urljoin(BASE_URL, filename), local_path, year))

    logger.info(f"Need to download {len(downloads)} files")

    # Download files concurrently
    downloaded_paths = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_DOWNLOAD_THREADS
    ) as executor:
        future_to_file = {
            executor.submit(download_file_with_retry, url, path, year): path
            for _, url, path, year in downloads
        }

        for future in concurrent.futures.as_completed(future_to_file):
            path = future_to_file[future]
            success = future.result()
            if success:
                downloaded_paths.append(path)
            else:
                logger.error(f"Failed to download {path}")

    return downloaded_paths
