"""Parquet-based metadata management for GHCNh stations."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import polars as pl

# Schema version - increment for breaking schema changes
METADATA_SCHEMA_VERSION = "2.0"

from weather_imputation.config.paths import (
    METADATA_PARQUET,
    METADATA_CLEANED_PARQUET,
    METADATA_CSV,
    METADATA_CLEANED_CSV,
    PROCESSED_DIR,
    STATION_LIST_PATH,
    get_station_year_path,
)
from weather_imputation.data.ghcnh_loader import (
    load_station_all_years,
    list_all_stations,
    list_available_years,
    get_station_list,
    filter_hourly_records,
    count_report_types,
    TIER1_VARIABLES,
    TIER2_VARIABLES,
)
from weather_imputation.data.stats import (
    compute_completeness,
    compute_variable_stats,
    compute_gap_analysis,
)

logger = logging.getLogger(__name__)


def compute_station_metadata(
    station_id: str,
    years: list[int] | None = None,
    filter_to_hourly: bool = True,
) -> dict[str, Any] | None:
    """Compute metadata for a single station.

    Args:
        station_id: Station identifier
        years: Optional list of years to include. If None, uses all available.
        filter_to_hourly: If True, filter to only hourly instrument records
            (excludes FM-12 SYNOP and other summary/aggregate records)

    Returns:
        Dictionary with station metadata, or None if no data found
    """
    available_years = list_available_years(station_id)
    if not available_years:
        return None

    if years is not None:
        available_years = [y for y in available_years if y in years]
        if not available_years:
            return None

    # Load all data for the station
    df_all = load_station_all_years(station_id, available_years)
    if df_all is None or len(df_all) == 0:
        return None

    # Count report types BEFORE filtering (for metadata)
    report_type_counts = count_report_types(df_all, "temperature_Report_Type")
    total_records_before_filter = len(df_all)

    # === GROUP 1: STATION IDENTIFIERS (from unfiltered data) ===
    metadata: dict[str, Any] = {
        "station_id": station_id,
        "country_code": station_id[:2] if len(station_id) >= 2 else "",
        "station_name": "",  # Will be filled below
        "latitude": None,    # Will be filled below
        "longitude": None,   # Will be filled below
        "elevation": None,   # Will be filled below
        # Placeholders for enrichment columns (filled by enrich_metadata_from_station_list)
        "state": None,
        "wmo_id": None,
        "icao_code": None,
    }

    # Extract station name from unfiltered data
    if "Station_name" in df_all.columns:
        names = df_all["Station_name"].unique().to_list()
        metadata["station_name"] = names[0] if names else ""

    # Location from unfiltered data (may vary across years)
    # Handle both uppercase and mixed-case column names
    for col_key, col_variants in [("latitude", ["LATITUDE", "Latitude"]), ("longitude", ["LONGITUDE", "Longitude"])]:
        for col in col_variants:
            if col in df_all.columns:
                # Use the most common non-null value
                values = df_all.filter(pl.col(col).is_not_null())[col]
                if len(values) > 0:
                    metadata[col_key] = values[0]
                break

    # Elevation from unfiltered data if available
    if "Elevation" in df_all.columns:
        elev_values = df_all.filter(pl.col("Elevation").is_not_null())["Elevation"]
        metadata["elevation"] = elev_values[0] if len(elev_values) > 0 else None

    # Filter to hourly instrument records if requested
    if filter_to_hourly:
        df = filter_hourly_records(df_all, "temperature_Report_Type")
    else:
        df = df_all

    # === GROUP 2: DATA SOURCE INFO ===
    metadata["report_type_counts"] = report_type_counts
    metadata["total_records_all_types"] = total_records_before_filter
    metadata["records_excluded_by_filter"] = total_records_before_filter - len(df)

    # If no records after filtering, return partial metadata with nulls/zeros
    # This allows us to still see station info and report type counts
    if len(df) == 0:
        metadata["first_observation"] = None
        metadata["last_observation"] = None
        metadata["years_available"] = available_years
        metadata["total_observation_count"] = 0
        metadata["year_counts"] = {}

        # Completeness - all 0%
        for var in TIER1_VARIABLES:
            metadata[f"{var}_completeness_pct"] = 0.0
        for var in TIER2_VARIABLES:
            metadata[f"{var}_completeness_pct"] = 0.0

        # Stats - all null
        metadata["temperature_stats"] = {"min": None, "max": None, "mean": None, "std": None}
        metadata["dew_point_stats"] = {"mean": None, "std": None}
        metadata["sea_level_pressure_stats"] = {"min": None, "max": None, "mean": None, "std": None}

        # Gap analysis - all null/zero
        metadata["gap_count_24h"] = 0
        metadata["max_gap_duration_hours"] = None
        metadata["avg_observation_interval_hours"] = None

        # Metadata
        metadata["metadata_computed_at"] = datetime.now()
        metadata["metadata_schema_version"] = METADATA_SCHEMA_VERSION

        return metadata

    # === GROUP 3: TEMPORAL COVERAGE ===
    # Handle multiple date formats:
    # - Old format: DATE column as string "2024-01-01T12:00:00" or datetime
    # - New format (2025+): Year, Month, Day, Hour, Minute as separate columns
    date_col_name = None

    if "DATE" in df.columns:
        date_col = df["DATE"]
        # Convert string dates to datetime if needed
        if date_col.dtype == pl.Utf8:
            df = df.with_columns(
                pl.col("DATE").str.to_datetime(format="%Y-%m-%dT%H:%M:%S", strict=False).alias("_date_parsed")
            )
            date_col_name = "_date_parsed"
        else:
            date_col_name = "DATE"

    # For files with Year/Month/Day/Hour/Minute columns (2025+ format),
    # create a datetime column from components if DATE is missing for those rows
    if all(col in df.columns for col in ["Year", "Month", "Day", "Hour", "Minute"]):
        # Create datetime from components for rows where _date_parsed is null
        df = df.with_columns(
            pl.datetime(
                pl.col("Year").cast(pl.Int32),
                pl.col("Month").cast(pl.Int32),
                pl.col("Day").cast(pl.Int32),
                pl.col("Hour").cast(pl.Int32),
                pl.col("Minute").cast(pl.Int32),
            ).alias("_date_from_components")
        )
        # Combine: use parsed DATE where available, otherwise use components
        if date_col_name:
            df = df.with_columns(
                pl.when(pl.col(date_col_name).is_null())
                .then(pl.col("_date_from_components"))
                .otherwise(pl.col(date_col_name))
                .alias("_date_unified")
            )
            date_col_name = "_date_unified"
        else:
            date_col_name = "_date_from_components"

    if date_col_name and date_col_name in df.columns:
        dates = df[date_col_name]
        metadata["first_observation"] = dates.min()
        metadata["last_observation"] = dates.max()
    else:
        metadata["first_observation"] = None
        metadata["last_observation"] = None

    metadata["years_available"] = available_years
    metadata["total_observation_count"] = len(df)

    # Year counts
    if date_col_name and date_col_name in df.columns:
        year_counts = (
            df.with_columns(pl.col(date_col_name).dt.year().alias("year"))
            .group_by("year")
            .agg(pl.count().alias("count"))
            .sort("year")
        )
        metadata["year_counts"] = {
            row["year"]: row["count"] for row in year_counts.iter_rows(named=True)
        }
    else:
        metadata["year_counts"] = {}

    # === GROUP 4: COMPLETENESS ===
    # Tier 1 completeness
    for var in TIER1_VARIABLES:
        qc_col = f"{var}_Quality_Code"
        pct = compute_completeness(df, var, qc_col if qc_col in df.columns else None)
        metadata[f"{var}_completeness_pct"] = pct

    # Tier 2 completeness
    for var in TIER2_VARIABLES:
        qc_col = f"{var}_Quality_Code"
        pct = compute_completeness(df, var, qc_col if qc_col in df.columns else None)
        metadata[f"{var}_completeness_pct"] = pct

    # === GROUP 5: STATISTICS (as JSON dicts) ===
    # Temperature statistics
    if "temperature" in df.columns:
        stats = compute_variable_stats(df, "temperature", "temperature_Quality_Code")
        metadata["temperature_stats"] = {
            "min": stats.min_val,
            "max": stats.max_val,
            "mean": stats.mean,
            "std": stats.std,
        }
    else:
        metadata["temperature_stats"] = {"min": None, "max": None, "mean": None, "std": None}

    # Dew point statistics
    if "dew_point_temperature" in df.columns:
        stats = compute_variable_stats(df, "dew_point_temperature", "dew_point_temperature_Quality_Code")
        metadata["dew_point_stats"] = {
            "mean": stats.mean,
            "std": stats.std,
        }
    else:
        metadata["dew_point_stats"] = {"mean": None, "std": None}

    # Sea level pressure statistics
    if "sea_level_pressure" in df.columns:
        stats = compute_variable_stats(df, "sea_level_pressure", "sea_level_pressure_Quality_Code")
        metadata["sea_level_pressure_stats"] = {
            "min": stats.min_val,
            "max": stats.max_val,
            "mean": stats.mean,
            "std": stats.std,
        }
    else:
        metadata["sea_level_pressure_stats"] = {"min": None, "max": None, "mean": None, "std": None}

    # === GROUP 6: GAP ANALYSIS ===
    # Use parsed date column if available
    if date_col_name:
        gap_analysis = compute_gap_analysis(df, date_column=date_col_name)
    else:
        gap_analysis = compute_gap_analysis(df)
    metadata["gap_count_24h"] = gap_analysis.gap_count_24h
    metadata["max_gap_duration_hours"] = gap_analysis.max_gap_hours
    metadata["avg_observation_interval_hours"] = gap_analysis.avg_observation_interval_hours

    # === GROUP 7: METADATA ===
    metadata["metadata_computed_at"] = datetime.now()
    metadata["metadata_schema_version"] = METADATA_SCHEMA_VERSION

    return metadata


def _process_station_for_thread(
    station_id: str,
    year_filter: list[int] | None,
) -> dict[str, Any] | None:
    """Thread-safe wrapper for compute_station_metadata.

    Handles JSON serialization of complex types before returning.
    Returns None on error (logged but not raised).
    """
    try:
        metadata = compute_station_metadata(station_id, year_filter)
        if metadata is None:
            return None

        # Convert complex types to JSON strings for Polars compatibility
        if "year_counts" in metadata and isinstance(metadata["year_counts"], dict):
            metadata["year_counts"] = json.dumps(metadata["year_counts"])
        if "years_available" in metadata and isinstance(metadata["years_available"], list):
            metadata["years_available"] = json.dumps(metadata["years_available"])
        if "report_type_counts" in metadata and isinstance(metadata["report_type_counts"], dict):
            metadata["report_type_counts"] = json.dumps(metadata["report_type_counts"])
        # Convert stats dicts to JSON strings
        for stats_key in ["temperature_stats", "dew_point_stats", "sea_level_pressure_stats"]:
            if stats_key in metadata and isinstance(metadata[stats_key], dict):
                metadata[stats_key] = json.dumps(metadata[stats_key])

        return metadata

    except Exception as e:
        logger.warning(f"Error computing metadata for {station_id}: {e}")
        return None


def _compute_metadata_parallel(
    stations: list[str],
    year_filter: list[int] | None = None,
    max_workers: int = 4,
    progress_callback: Any = None,
) -> list[dict[str, Any]]:
    """Compute metadata for stations in parallel using ThreadPoolExecutor.

    Args:
        stations: List of station IDs to process
        year_filter: Optional list of years to include
        max_workers: Maximum number of parallel worker threads.
            Use 1 for sequential processing (backward compatibility).
        progress_callback: Optional callback(current, total, station_id)
            Called from main thread after each station completes.

    Returns:
        List of metadata dictionaries (None results filtered out)
    """
    if not stations:
        return []

    # For single worker, use simple sequential processing (backward compatibility)
    if max_workers == 1:
        metadata_list: list[dict[str, Any]] = []
        for i, station_id in enumerate(stations):
            if progress_callback:
                progress_callback(i, len(stations), station_id)
            result = _process_station_for_thread(station_id, year_filter)
            if result is not None:
                metadata_list.append(result)
        return metadata_list

    # Parallel processing
    metadata_list = []
    results_lock = Lock()
    completed_count = 0
    completed_lock = Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_station = {
            executor.submit(_process_station_for_thread, station_id, year_filter): station_id
            for station_id in stations
        }

        # Process completions from main thread (safe for Rich progress)
        for future in as_completed(future_to_station):
            station_id = future_to_station[future]
            try:
                result = future.result()
                if result is not None:
                    with results_lock:
                        metadata_list.append(result)
            except Exception as e:
                logger.warning(f"Exception processing {station_id}: {e}")

            with completed_lock:
                completed_count += 1
                current = completed_count

            if progress_callback:
                progress_callback(current, len(stations), station_id)

    return metadata_list


def compute_all_metadata(
    year_filter: list[int] | None = None,
    station_filter: list[str] | None = None,
    progress_callback: Any = None,
    max_workers: int = 1,
) -> pl.DataFrame:
    """Compute metadata for all stations.

    Args:
        year_filter: Optional list of years to include
        station_filter: Optional list of station IDs to process
        progress_callback: Optional callback for progress updates
        max_workers: Number of parallel workers (default 1 for sequential)

    Returns:
        Polars DataFrame with metadata for all stations
    """
    stations = station_filter or list_all_stations()

    if not stations:
        logger.warning("No stations found to compute metadata for")
        return pl.DataFrame()

    metadata_list = _compute_metadata_parallel(
        stations=stations,
        year_filter=year_filter,
        max_workers=max_workers,
        progress_callback=progress_callback,
    )

    if not metadata_list:
        return pl.DataFrame()

    return pl.DataFrame(metadata_list)


def load_metadata(cleaned: bool = False) -> pl.DataFrame | None:
    """Load station metadata from parquet file.

    Args:
        cleaned: If True, load cleaned metadata; otherwise load raw

    Returns:
        Polars DataFrame with metadata, or None if file doesn't exist
    """
    path = METADATA_CLEANED_PARQUET if cleaned else METADATA_PARQUET

    if not path.exists():
        logger.info(f"Metadata file not found: {path}")
        return None

    try:
        df = pl.read_parquet(path)

        # Check schema version
        if "metadata_schema_version" not in df.columns:
            logger.warning(
                "Metadata file uses old schema (no version field). "
                "Consider recomputing with: uv run python src/scripts/compute_metadata.py compute --force"
            )
        elif len(df) > 0:
            file_version = df["metadata_schema_version"][0]
            if file_version != METADATA_SCHEMA_VERSION:
                logger.warning(
                    f"Metadata schema version mismatch: file has {file_version}, "
                    f"current is {METADATA_SCHEMA_VERSION}. Consider recomputing."
                )

        return df
    except Exception as e:
        logger.error(f"Error reading metadata from {path}: {e}")
        return None


def _reorder_metadata_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Reorder metadata columns to standard schema order.

    Order:
    1. Station identifiers: station_id, country_code, station_name, latitude, longitude, elevation, state, wmo_id, icao_code
    2. Data source: report_type_counts, total_records_all_types, records_excluded_by_filter
    3. Temporal: first_observation, last_observation, years_available, total_observation_count, year_counts
    4. Completeness: *_completeness_pct columns
    5. Statistics: temperature_stats, dew_point_stats, sea_level_pressure_stats
    6. Gap analysis: gap_count_24h, max_gap_duration_hours, avg_observation_interval_hours
    7. Metadata: metadata_computed_at, metadata_schema_version, coord_valid (if present)
    """
    preferred_order = [
        # Group 1: Station identifiers
        "station_id", "country_code", "station_name",
        "latitude", "longitude", "elevation",
        "state", "wmo_id", "icao_code",
        # Group 2: Data source info
        "report_type_counts", "total_records_all_types", "records_excluded_by_filter",
        # Group 3: Temporal coverage
        "first_observation", "last_observation", "years_available",
        "total_observation_count", "year_counts",
    ]

    # Build final column order
    final_order = []
    remaining = set(df.columns)

    # Add preferred columns first (in order)
    for col in preferred_order:
        if col in remaining:
            final_order.append(col)
            remaining.remove(col)

    # Add completeness columns (sorted)
    completeness_cols = sorted([c for c in remaining if c.endswith("_completeness_pct")])
    final_order.extend(completeness_cols)
    remaining -= set(completeness_cols)

    # Add stats columns
    stats_cols = ["temperature_stats", "dew_point_stats", "sea_level_pressure_stats"]
    for col in stats_cols:
        if col in remaining:
            final_order.append(col)
            remaining.remove(col)

    # Add gap analysis columns
    gap_cols = ["gap_count_24h", "max_gap_duration_hours", "avg_observation_interval_hours"]
    for col in gap_cols:
        if col in remaining:
            final_order.append(col)
            remaining.remove(col)

    # Add metadata columns at end
    meta_cols = ["metadata_computed_at", "metadata_schema_version", "coord_valid"]
    for col in meta_cols:
        if col in remaining:
            final_order.append(col)
            remaining.remove(col)

    # Add any remaining columns (for forward compatibility)
    final_order.extend(sorted(remaining))

    return df.select(final_order)


def save_metadata(
    df: pl.DataFrame,
    cleaned: bool = False,
    export_csv: bool = True,
    csv_path: Path | None = None,
) -> Path | None:
    """Save station metadata to parquet (and optionally CSV).

    Args:
        df: DataFrame with station metadata
        cleaned: If True, save as cleaned metadata; otherwise as raw
        export_csv: If True, also export to CSV
        csv_path: Optional custom path for CSV export. If None, uses default
            based on 'cleaned' flag (metadata.csv or metadata_cleaned.csv)

    Returns:
        Path to the CSV file if exported, None otherwise
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    path = METADATA_CLEANED_PARQUET if cleaned else METADATA_PARQUET

    # Convert year_counts dict to JSON string for parquet storage
    if "year_counts" in df.columns:
        df = df.with_columns(
            pl.col("year_counts").map_elements(
                lambda x: json.dumps(x) if isinstance(x, dict) else x,
                return_dtype=pl.Utf8,
            ).alias("year_counts")
        )

    # Convert years_available list to JSON string
    if "years_available" in df.columns:
        df = df.with_columns(
            pl.col("years_available").map_elements(
                lambda x: json.dumps(x) if isinstance(x, list) else x,
                return_dtype=pl.Utf8,
            ).alias("years_available")
        )

    # Convert report_type_counts dict to JSON string
    if "report_type_counts" in df.columns:
        df = df.with_columns(
            pl.col("report_type_counts").map_elements(
                lambda x: json.dumps(x) if isinstance(x, dict) else x,
                return_dtype=pl.Utf8,
            ).alias("report_type_counts")
        )

    # Convert stats dicts to JSON strings
    for stats_col in ["temperature_stats", "dew_point_stats", "sea_level_pressure_stats"]:
        if stats_col in df.columns:
            df = df.with_columns(
                pl.col(stats_col).map_elements(
                    lambda x: json.dumps(x) if isinstance(x, dict) else x,
                    return_dtype=pl.Utf8,
                ).alias(stats_col)
            )

    # Reorder columns to standard schema order
    df = _reorder_metadata_columns(df)

    df.write_parquet(path)
    logger.info(f"Saved metadata to {path}")

    if export_csv:
        # Use custom path if provided, otherwise select based on cleaned flag
        output_csv = csv_path or (METADATA_CLEANED_CSV if cleaned else METADATA_CSV)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(output_csv)
        logger.info(f"Exported metadata to {output_csv}")
        return output_csv

    return None


def get_station_file_mtime(station_id: str, years: list[int] | None = None) -> datetime | None:
    """Get the latest modification time of a station's parquet files.

    Args:
        station_id: Station identifier
        years: Optional list of years to check. If None, checks all available years.

    Returns:
        Latest modification time across all files, or None if no files exist
    """
    if years is None:
        years = list_available_years(station_id)

    if not years:
        return None

    latest_mtime: datetime | None = None

    for year in years:
        path = get_station_year_path(station_id, year)
        if path.exists():
            file_mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if latest_mtime is None or file_mtime > latest_mtime:
                latest_mtime = file_mtime

    return latest_mtime


def get_stations_needing_update(
    existing_metadata: pl.DataFrame | None,
    year_filter: list[int] | None = None,
    station_filter: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Identify stations that need metadata recomputation.

    A station needs update if:
    - It has no existing metadata
    - Its parquet files have been modified since metadata was computed
    - It has new year files available

    Args:
        existing_metadata: Current metadata DataFrame (or None if no metadata exists)
        year_filter: Optional list of years to consider
        station_filter: Optional list of station IDs to consider

    Returns:
        Tuple of (stations_to_update, stations_unchanged)
    """
    all_stations = station_filter or list_all_stations()

    if existing_metadata is None or len(existing_metadata) == 0:
        # No existing metadata - all stations need computation
        return all_stations, []

    # Build lookup of existing metadata
    existing_lookup: dict[str, datetime | None] = {}
    for row in existing_metadata.iter_rows(named=True):
        station_id = row.get("station_id")
        computed_at = row.get("metadata_computed_at")
        if station_id:
            existing_lookup[station_id] = computed_at

    stations_to_update: list[str] = []
    stations_unchanged: list[str] = []

    for station_id in all_stations:
        if station_id not in existing_lookup:
            # New station - needs computation
            stations_to_update.append(station_id)
            continue

        computed_at = existing_lookup[station_id]
        if computed_at is None:
            # No timestamp recorded - recompute
            stations_to_update.append(station_id)
            continue

        # Check if any files have been modified since metadata was computed
        file_mtime = get_station_file_mtime(station_id, year_filter)

        if file_mtime is None:
            # No files found - station data may have been removed
            # Keep existing metadata but don't update
            stations_unchanged.append(station_id)
            continue

        if file_mtime > computed_at:
            # Files have been modified since metadata was computed
            stations_to_update.append(station_id)
        else:
            stations_unchanged.append(station_id)

    return stations_to_update, stations_unchanged


def compute_all_metadata_incremental(
    year_filter: list[int] | None = None,
    station_filter: list[str] | None = None,
    progress_callback: Any = None,
    max_workers: int = 1,
) -> tuple[pl.DataFrame, int, int]:
    """Compute metadata incrementally, only updating stations with changed files.

    Args:
        year_filter: Optional list of years to include
        station_filter: Optional list of station IDs to process
        progress_callback: Optional callback for progress updates
        max_workers: Number of parallel workers (default 1 for sequential)

    Returns:
        Tuple of (merged_metadata_df, stations_updated, stations_unchanged)
    """
    # Load existing metadata
    existing_metadata = load_metadata(cleaned=False)

    # Identify which stations need update
    stations_to_update, stations_unchanged = get_stations_needing_update(
        existing_metadata, year_filter, station_filter
    )

    logger.info(
        f"Incremental update: {len(stations_to_update)} stations to update, "
        f"{len(stations_unchanged)} unchanged"
    )

    if not stations_to_update:
        # Nothing to update - return existing metadata
        if existing_metadata is not None:
            return existing_metadata, 0, len(stations_unchanged)
        return pl.DataFrame(), 0, 0

    # Compute metadata for stations that need updates (parallel or sequential)
    new_metadata_list = _compute_metadata_parallel(
        stations=stations_to_update,
        year_filter=year_filter,
        max_workers=max_workers,
        progress_callback=progress_callback,
    )

    if not new_metadata_list:
        # No new metadata computed - return existing
        if existing_metadata is not None:
            return existing_metadata, 0, len(stations_unchanged)
        return pl.DataFrame(), 0, 0

    new_metadata_df = pl.DataFrame(new_metadata_list)

    # Merge with existing metadata
    if existing_metadata is not None and len(existing_metadata) > 0:
        # Remove stations that we're updating from existing metadata
        updated_station_ids = [m["station_id"] for m in new_metadata_list]
        existing_kept = existing_metadata.filter(
            ~pl.col("station_id").is_in(updated_station_ids)
        )

        # Concatenate existing (kept) with new
        if len(existing_kept) > 0:
            # Use diagonal_relaxed to handle any schema differences
            merged = pl.concat([existing_kept, new_metadata_df], how="diagonal_relaxed")
        else:
            merged = new_metadata_df
    else:
        merged = new_metadata_df

    return merged, len(new_metadata_list), len(stations_unchanged)


def download_station_list(dest_path: Path | None = None) -> bool:
    """Download the GHCNh station list from NOAA.

    Args:
        dest_path: Destination path. Defaults to STATION_LIST_PATH.

    Returns:
        True if download successful, False otherwise
    """
    import requests

    path = dest_path or STATION_LIST_PATH
    url = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh-station-list.csv"

    try:
        logger.info(f"Downloading station list from {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
        logger.info(f"Saved station list to {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download station list: {e}")
        return False


def enrich_metadata_from_station_list(
    metadata: pl.DataFrame,
    station_list_path: Path | None = None,
    download_if_missing: bool = True,
) -> pl.DataFrame:
    """Enrich metadata with information from the GHCNh station list.

    Fills in missing latitude, longitude, elevation, and other fields
    from the official station list.

    Args:
        metadata: DataFrame with station metadata
        station_list_path: Path to station list (defaults to standard location)
        download_if_missing: If True, download station list if not found locally

    Returns:
        Enriched metadata DataFrame
    """
    path = station_list_path or STATION_LIST_PATH

    # Download if missing
    if not path.exists() and download_if_missing:
        if not download_station_list(path):
            logger.warning("Could not download station list for enrichment")
            return metadata

    station_list = get_station_list(path)
    if station_list is None:
        logger.warning("Could not load station list for enrichment")
        return metadata

    # Clean up column names (Windows line endings can add \r to last column)
    station_list = station_list.rename({
        col: col.strip() for col in station_list.columns
    })

    # Rename columns to match our metadata schema
    # Station list columns: GHCN_ID, LATITUDE, LONGITUDE, ELEVATION, STATE, NAME, GSN, (US)HCN_(US)CRN, WMO_ID, ICAO
    rename_map = {
        "GHCN_ID": "station_id",
        "LATITUDE": "inv_latitude",
        "LONGITUDE": "inv_longitude",
        "ELEVATION": "inv_elevation",
        "STATE": "state",
        "NAME": "inv_station_name",
        "WMO_ID": "wmo_id",
        "ICAO": "icao_code",
    }
    # Only rename columns that exist
    rename_map = {k: v for k, v in rename_map.items() if k in station_list.columns}
    station_list = station_list.rename(rename_map)

    # Strip whitespace from string columns (ICAO is the last column and can have \r from Windows line endings)
    for col in station_list.columns:
        if station_list[col].dtype == pl.Utf8:
            station_list = station_list.with_columns(
                pl.col(col).str.strip_chars().alias(col)
            )

    # Select only the columns we need
    enrichment_cols = [
        "station_id", "inv_latitude", "inv_longitude", "inv_elevation",
        "state", "inv_station_name", "wmo_id", "icao_code"
    ]
    available_cols = [c for c in enrichment_cols if c in station_list.columns]
    station_list = station_list.select(available_cols)

    # Join with metadata (use suffix for columns that already exist in metadata)
    enriched = metadata.join(station_list, on="station_id", how="left", suffix="_inv")

    # Fill in missing values from inventory
    if "inv_latitude" in enriched.columns:
        enriched = enriched.with_columns(
            pl.when(pl.col("latitude").is_null())
            .then(pl.col("inv_latitude"))
            .otherwise(pl.col("latitude"))
            .alias("latitude")
        ).drop("inv_latitude")

    if "inv_longitude" in enriched.columns:
        enriched = enriched.with_columns(
            pl.when(pl.col("longitude").is_null())
            .then(pl.col("inv_longitude"))
            .otherwise(pl.col("longitude"))
            .alias("longitude")
        ).drop("inv_longitude")

    if "inv_elevation" in enriched.columns:
        enriched = enriched.with_columns(
            pl.when(pl.col("elevation").is_null())
            .then(pl.col("inv_elevation"))
            .otherwise(pl.col("elevation"))
            .alias("elevation")
        ).drop("inv_elevation")

    if "inv_station_name" in enriched.columns:
        enriched = enriched.with_columns(
            pl.when(pl.col("station_name").is_null() | (pl.col("station_name") == ""))
            .then(pl.col("inv_station_name"))
            .otherwise(pl.col("station_name"))
            .alias("station_name")
        ).drop("inv_station_name")

    # Fill in state, wmo_id, icao_code from inventory (these have _inv suffix due to join)
    for col in ["state", "wmo_id", "icao_code"]:
        inv_col = f"{col}_inv"
        if inv_col in enriched.columns:
            enriched = enriched.with_columns(
                pl.when(pl.col(col).is_null() | (pl.col(col) == ""))
                .then(pl.col(inv_col))
                .otherwise(pl.col(col))
                .alias(col)
            ).drop(inv_col)

    logger.info(f"Enriched metadata from station list ({len(station_list)} stations in inventory)")
    return enriched
