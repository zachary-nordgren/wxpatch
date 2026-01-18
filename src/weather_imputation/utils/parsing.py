"""Parsing utilities for command-line arguments and filters."""

import logging

logger = logging.getLogger(__name__)


def parse_year_filter(year_filter_str: str | None) -> list[int]:
    """Parse a year filter string into a list of years.

    Supports individual years and ranges:
    - Individual years: "2010,2012,2015"
    - Year ranges: "2012:2015" (inclusive of both 2012 and 2015)
    - Mixed: "2010,2012:2015,2017:2022,2024"

    Args:
        year_filter_str: Comma-separated years and/or ranges, or None

    Returns:
        Sorted list of years as integers

    Examples:
        >>> parse_year_filter("2020,2021,2022")
        [2020, 2021, 2022]
        >>> parse_year_filter("2018:2020")
        [2018, 2019, 2020]
        >>> parse_year_filter("2010,2015:2017,2020")
        [2010, 2015, 2016, 2017, 2020]
        >>> parse_year_filter(None)
        []
    """
    if not year_filter_str:
        return []

    years: set[int] = set()

    # Split by comma
    parts = year_filter_str.split(",")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            # Handle range
            range_parts = part.split(":")
            if len(range_parts) != 2:
                logger.warning(f"Invalid year range format: {part}. Skipping.")
                continue

            try:
                start = int(range_parts[0].strip())
                end = int(range_parts[1].strip())

                if start > end:
                    logger.warning(f"Invalid year range: {part} (start > end). Skipping.")
                    continue

                # Add all years in the range (inclusive)
                years.update(range(start, end + 1))
            except ValueError:
                logger.warning(f"Invalid year range: {part}. Skipping.")
                continue
        else:
            # Handle individual year
            try:
                years.add(int(part))
            except ValueError:
                logger.warning(f"Invalid year: {part}. Skipping.")
                continue

    return sorted(years)


def parse_station_filter(station_filter_str: str | None) -> list[str]:
    """Parse a station filter string into a list of station IDs.

    Args:
        station_filter_str: Comma-separated station IDs, or None

    Returns:
        List of station IDs (uppercase, stripped)

    Examples:
        >>> parse_station_filter("USW00003046,USW00012345")
        ['USW00003046', 'USW00012345']
        >>> parse_station_filter(None)
        []
    """
    if not station_filter_str:
        return []

    stations = []
    for station in station_filter_str.split(","):
        station = station.strip().upper()
        if station:
            stations.append(station)

    return stations
