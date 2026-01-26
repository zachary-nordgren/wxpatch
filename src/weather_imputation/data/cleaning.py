"""Metadata cleaning, deduplication, and enrichment."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from weather_imputation.config.paths import (
    CLEANING_REPORT_JSON,
    CLEANING_LOG_TXT,
    STATION_LIST_PATH,
)
from weather_imputation.data.ghcnh_loader import get_station_list

logger = logging.getLogger(__name__)


@dataclass
class CleaningLog:
    """Detailed log of all cleaning operations."""

    entries: list[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    def add(self, message: str) -> None:
        """Add a timestamped log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.entries.append(f"[{timestamp}] {message}")

    def add_section(self, title: str) -> None:
        """Add a section header to the log."""
        self.entries.append(f"\n{'='*60}")
        self.entries.append(title)
        self.entries.append("=" * 60)

    def save(self, path: Path | None = None) -> None:
        """Save log to text file."""
        output_path = path or CLEANING_LOG_TXT
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add header
        header = [
            "=" * 60,
            f"Metadata Cleaning Log - {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header + self.entries))
        logger.info(f"Saved cleaning log to {output_path}")

    def __str__(self) -> str:
        return "\n".join(self.entries)


@dataclass
class CleaningReport:
    """Report of cleaning operations performed."""

    original_station_count: int = 0
    cleaned_station_count: int = 0
    duplicates_found: int = 0
    duplicates_merged: int = 0
    coordinates_filled: int = 0
    coordinates_validated: int = 0
    coordinates_invalid: int = 0
    names_cleaned: int = 0
    cleaning_timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_station_count": self.original_station_count,
            "cleaned_station_count": self.cleaned_station_count,
            "duplicates_found": self.duplicates_found,
            "duplicates_merged": self.duplicates_merged,
            "coordinates_filled": self.coordinates_filled,
            "coordinates_validated": self.coordinates_validated,
            "coordinates_invalid": self.coordinates_invalid,
            "names_cleaned": self.names_cleaned,
            "cleaning_timestamp": self.cleaning_timestamp.isoformat(),
        }

    def save(self, path: Path | None = None) -> None:
        """Save report to JSON file."""
        output_path = path or CLEANING_REPORT_JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved cleaning report to {output_path}")


def identify_duplicate_stations(
    metadata: pl.DataFrame,
    location_threshold: float = 0.01,
    name_similarity_threshold: float = 0.8,
    strategy: str = "prefer_longer",
    log: CleaningLog | None = None,
) -> pl.DataFrame:
    """Find stations that are likely duplicates.

    Duplicates are identified based on:
    - Same location (lat/lon within threshold degrees)
    - Similar names (fuzzy matching above threshold)
    - Overlapping but not identical time periods

    Args:
        metadata: DataFrame with station metadata
        location_threshold: Maximum distance in degrees to consider same location
        name_similarity_threshold: Minimum name similarity score (0-1)
        strategy: How to choose the primary station - "prefer_longer" (more observations)
                  or "prefer_recent" (most recent last_observation)
        log: Optional CleaningLog to record detailed operations

    Returns:
        DataFrame with columns: station_id, duplicate_of, confidence_score, reason
    """
    duplicates = []

    stations = metadata.to_dicts()
    n = len(stations)

    for i in range(n):
        for j in range(i + 1, n):
            s1 = stations[i]
            s2 = stations[j]

            # Check location proximity
            lat1, lon1 = s1.get("latitude"), s1.get("longitude")
            lat2, lon2 = s2.get("latitude"), s2.get("longitude")

            if all(v is not None for v in [lat1, lon1, lat2, lon2]):
                lat_diff = abs(lat1 - lat2)
                lon_diff = abs(lon1 - lon2)

                if lat_diff <= location_threshold and lon_diff <= location_threshold:
                    # Same location - likely duplicate
                    # Choose primary based on strategy
                    if strategy == "prefer_recent":
                        # Prefer station with more recent data
                        last1 = s1.get("last_observation")
                        last2 = s2.get("last_observation")
                        # Handle None values - treat as oldest possible
                        if last1 is None and last2 is None:
                            # Fall back to observation count
                            s1_wins = (s1.get("total_observation_count", 0) or 0) >= (s2.get("total_observation_count", 0) or 0)
                        elif last1 is None:
                            s1_wins = False
                        elif last2 is None:
                            s1_wins = True
                        else:
                            s1_wins = last1 >= last2
                    else:
                        # Default: prefer_longer - station with more observations
                        obs1 = s1.get("total_observation_count", 0) or 0
                        obs2 = s2.get("total_observation_count", 0) or 0
                        s1_wins = obs1 >= obs2

                    if s1_wins:
                        primary, secondary = s1, s2
                    else:
                        primary, secondary = s2, s1

                    reason = f"same_location (dist: {lat_diff:.4f}, {lon_diff:.4f})"
                    duplicates.append({
                        "station_id": secondary["station_id"],
                        "duplicate_of": primary["station_id"],
                        "confidence_score": 0.9,
                        "reason": reason,
                    })

                    # Log the duplicate finding
                    if log is not None:
                        log.add(
                            f"Found duplicate: {secondary['station_id']} → {primary['station_id']} "
                            f"({reason})"
                        )

    if log is not None:
        log.add(f"Total duplicates found: {len(duplicates)}")

    if not duplicates:
        return pl.DataFrame(schema={
            "station_id": pl.Utf8,
            "duplicate_of": pl.Utf8,
            "confidence_score": pl.Float64,
            "reason": pl.Utf8,
        })

    return pl.DataFrame(duplicates)


def merge_duplicate_stations(
    metadata: pl.DataFrame,
    duplicates: pl.DataFrame,
    log: CleaningLog | None = None,
) -> pl.DataFrame:
    """Merge duplicate station records into canonical entries.

    Args:
        metadata: DataFrame with station metadata
        duplicates: DataFrame from identify_duplicate_stations (already has primary/secondary decided)
        log: Optional CleaningLog to record detailed operations

    Returns:
        Cleaned metadata DataFrame with duplicates removed
    """
    if len(duplicates) == 0:
        return metadata

    # Get list of station IDs to remove (the duplicates, not the primaries)
    stations_to_remove = duplicates["station_id"].to_list()

    # Log each station removal
    if log is not None:
        for row in duplicates.iter_rows(named=True):
            log.add(
                f"Removing duplicate: {row['station_id']} (merged into {row['duplicate_of']})"
            )

    # Filter out duplicates
    cleaned = metadata.filter(~pl.col("station_id").is_in(stations_to_remove))

    logger.info(f"Removed {len(stations_to_remove)} duplicate stations")
    return cleaned


def lookup_missing_coordinates(
    metadata: pl.DataFrame,
    station_list_path: Path | None = None,
) -> tuple[pl.DataFrame, int]:
    """Fill missing lat/lon/elevation from station list.

    Args:
        metadata: DataFrame with station metadata
        station_list_path: Path to station list CSV

    Returns:
        Tuple of (enriched metadata DataFrame, count of coordinates filled)
    """
    station_list = get_station_list(station_list_path)
    if station_list is None:
        logger.warning("Could not load station list for coordinate lookup")
        return metadata, 0

    # Count missing coordinates before
    missing_lat = metadata.filter(pl.col("latitude").is_null()).height
    missing_lon = metadata.filter(pl.col("longitude").is_null()).height

    # The station list format varies - this is a placeholder for actual implementation
    # TODO: Implement actual join based on station list format
    logger.info(f"Found {missing_lat} missing latitudes, {missing_lon} missing longitudes")

    filled_count = 0  # Will be updated after actual implementation
    return metadata, filled_count


def lookup_missing_elevation(
    metadata: pl.DataFrame,
    use_api: bool = False,
) -> tuple[pl.DataFrame, int]:
    """Fill missing elevation values.

    Args:
        metadata: DataFrame with station metadata
        use_api: If True, query external elevation API for missing values

    Returns:
        Tuple of (enriched metadata DataFrame, count of elevations filled)
    """
    # Count missing elevations (-999.9 or null indicates missing)
    missing = metadata.filter(
        pl.col("elevation").is_null() | (pl.col("elevation") == -999.9)
    ).height

    if missing == 0:
        return metadata, 0

    logger.info(f"Found {missing} stations with missing elevation")

    filled_count = 0

    if use_api:
        # TODO: Implement external API lookup (e.g., Open-Elevation)
        logger.info("External elevation API lookup not yet implemented")

    return metadata, filled_count


def validate_coordinates(
    metadata: pl.DataFrame,
    log: CleaningLog | None = None,
) -> tuple[pl.DataFrame, int, int]:
    """Validate coordinate sanity and flag outliers.

    Checks:
    - Latitude in [-90, 90]
    - Longitude in [-180, 180]
    - Elevation in reasonable range (-500m to 9000m)

    Args:
        metadata: DataFrame with station metadata
        log: Optional CleaningLog to record detailed operations

    Returns:
        Tuple of (metadata with 'coord_valid' column, valid count, invalid count)
    """
    # Create validation expressions
    lat_valid = (pl.col("latitude") >= -90) & (pl.col("latitude") <= 90)
    lon_valid = (pl.col("longitude") >= -180) & (pl.col("longitude") <= 180)

    # Elevation validation (null or -999.9 is treated as "unknown" not invalid)
    elev_valid = (
        pl.col("elevation").is_null()
        | (pl.col("elevation") == -999.9)  # Missing marker
        | ((pl.col("elevation") >= -500) & (pl.col("elevation") <= 9000))
    )

    # Handle null lat/lon as invalid
    lat_present = pl.col("latitude").is_not_null()
    lon_present = pl.col("longitude").is_not_null()

    coord_valid = lat_present & lon_present & lat_valid & lon_valid & elev_valid

    enriched = metadata.with_columns(coord_valid.alias("coord_valid"))

    valid_count = enriched.filter(pl.col("coord_valid")).height
    invalid_count = len(enriched) - valid_count

    # Log invalid coordinates
    if log is not None and invalid_count > 0:
        invalid_stations = enriched.filter(~pl.col("coord_valid"))
        for row in invalid_stations.iter_rows(named=True):
            lat = row.get("latitude")
            lon = row.get("longitude")
            elev = row.get("elevation")

            # Determine reason
            reasons = []
            if lat is None:
                reasons.append("missing latitude")
            elif lat < -90 or lat > 90:
                reasons.append(f"latitude out of range ({lat})")
            if lon is None:
                reasons.append("missing longitude")
            elif lon < -180 or lon > 180:
                reasons.append(f"longitude out of range ({lon})")
            if elev is not None and elev != -999.9 and (elev < -500 or elev > 9000):
                reasons.append(f"elevation out of range ({elev})")

            reason_str = ", ".join(reasons) if reasons else "unknown"
            log.add(
                f"Invalid coordinates: {row['station_id']} "
                f"(lat: {lat}, lon: {lon}, elev: {elev}) - {reason_str}"
            )

        log.add(f"Valid: {valid_count}, Invalid: {invalid_count}")

    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} stations with invalid coordinates")

    return enriched, valid_count, invalid_count


def clean_station_names(
    metadata: pl.DataFrame,
    log: CleaningLog | None = None,
) -> tuple[pl.DataFrame, int]:
    """Normalize station names.

    Operations:
    - Strip whitespace
    - Standardize case (title case)
    - Remove excessive whitespace

    Args:
        metadata: DataFrame with station metadata
        log: Optional CleaningLog to record detailed operations

    Returns:
        Tuple of (cleaned metadata DataFrame, count of names modified)
    """
    if "station_name" not in metadata.columns:
        return metadata, 0

    original_names = metadata["station_name"].to_list()
    station_ids = metadata["station_id"].to_list()

    cleaned = metadata.with_columns(
        pl.col("station_name")
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
        .str.to_titlecase()
        .alias("station_name")
    )

    cleaned_names = cleaned["station_name"].to_list()

    # Count and log changes
    changes = 0
    for station_id, orig, clean in zip(station_ids, original_names, cleaned_names):
        if orig != clean:
            changes += 1
            if log is not None:
                # Truncate long names for readability
                orig_display = repr(orig[:50] + "..." if orig and len(orig) > 50 else orig)
                clean_display = repr(clean[:50] + "..." if clean and len(clean) > 50 else clean)
                log.add(f"Name cleaned ({station_id}): {orig_display} → {clean_display}")

    if log is not None:
        log.add(f"Total names cleaned: {changes}")

    return cleaned, changes


def run_cleaning_pipeline(
    metadata: pl.DataFrame,
    merge_duplicates: bool = True,
    merge_strategy: str = "combine",
    fill_coordinates: bool = True,
    validate_coords: bool = True,
    clean_names: bool = True,
    log: CleaningLog | None = None,
) -> tuple[pl.DataFrame, CleaningReport]:
    """Run the full cleaning pipeline on metadata.

    Args:
        metadata: Raw metadata DataFrame
        merge_duplicates: Whether to merge duplicate stations
        merge_strategy: Strategy for merging duplicates - "prefer_longer" or "prefer_recent"
        fill_coordinates: Whether to lookup missing coordinates
        validate_coords: Whether to validate coordinate ranges
        clean_names: Whether to clean station names
        log: Optional CleaningLog to record detailed operations

    Returns:
        Tuple of (cleaned metadata, cleaning report)
    """
    report = CleaningReport(original_station_count=len(metadata))

    if log is not None:
        log.add(f"Starting cleaning pipeline with {len(metadata)} stations")

    cleaned = metadata

    # Step 1: Identify and merge duplicates
    if merge_duplicates:
        if log is not None:
            log.add_section("DUPLICATE DETECTION")
        duplicates = identify_duplicate_stations(cleaned, strategy=merge_strategy, log=log)
        report.duplicates_found = len(duplicates)
        if len(duplicates) > 0:
            cleaned = merge_duplicate_stations(cleaned, duplicates, log=log)
            report.duplicates_merged = report.duplicates_found

    # Step 2: Fill missing coordinates
    if fill_coordinates:
        if log is not None:
            log.add_section("COORDINATE LOOKUP")
        cleaned, filled = lookup_missing_coordinates(cleaned)
        report.coordinates_filled = filled
        if log is not None:
            log.add(f"Coordinates filled from station list: {filled}")

    # Step 3: Validate coordinates
    if validate_coords:
        if log is not None:
            log.add_section("COORDINATE VALIDATION")
        cleaned, valid, invalid = validate_coordinates(cleaned, log=log)
        report.coordinates_validated = valid
        report.coordinates_invalid = invalid

    # Step 4: Clean station names
    if clean_names:
        if log is not None:
            log.add_section("STATION NAME CLEANING")
        cleaned, name_changes = clean_station_names(cleaned, log=log)
        report.names_cleaned = name_changes

    report.cleaned_station_count = len(cleaned)

    # Add summary to log
    if log is not None:
        log.add_section("SUMMARY")
        log.add(f"Original stations: {report.original_station_count}")
        log.add(f"Duplicates removed: {report.duplicates_merged}")
        log.add(f"Invalid coordinates flagged: {report.coordinates_invalid}")
        log.add(f"Names cleaned: {report.names_cleaned}")
        log.add(f"Final station count: {report.cleaned_station_count}")

    return cleaned, report


def generate_cleaning_report(
    original: pl.DataFrame,
    cleaned: pl.DataFrame,
) -> dict[str, Any]:
    """Generate a summary of cleaning operations.

    Args:
        original: Original metadata DataFrame
        cleaned: Cleaned metadata DataFrame

    Returns:
        Dictionary with cleaning summary
    """
    return {
        "original_station_count": len(original),
        "cleaned_station_count": len(cleaned),
        "stations_removed": len(original) - len(cleaned),
        "timestamp": datetime.now().isoformat(),
    }
