"""GHCNh parquet data loading utilities."""

import logging
from pathlib import Path

import polars as pl

from weather_imputation.config.paths import (
    GHCNH_RAW_DIR,
    STATION_LIST_PATH,
    get_station_year_path,
)

logger = logging.getLogger(__name__)

# GHCNh column definitions (from documentation)
# Primary columns always present
PRIMARY_COLUMNS = ["STATION", "Station_name", "DATE", "LATITUDE", "LONGITUDE"]

# Tier 1 weather variables (most commonly available)
TIER1_VARIABLES = [
    "temperature",
    "dew_point_temperature",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
    "relative_humidity",
]

# Tier 2 weather variables (less commonly available)
TIER2_VARIABLES = [
    "visibility",
    "wind_gust",
    "precipitation",
]

# All weather variables
ALL_WEATHER_VARIABLES = TIER1_VARIABLES + TIER2_VARIABLES

# Report types that represent hourly instrument readings (include these)
# Based on GHCNh documentation Section VII, Table 4
HOURLY_REPORT_TYPES = {
    # METAR/Aviation reports (hourly or sub-hourly instrument readings)
    "FM15-METAR",  # FM15: Aviation routine weather (standard hourly)
    "FM-15",
    "FM16-SPECI",  # FM16: Aviation special weather (triggered by changing conditions)
    "FM-16",
    "AUTO",        # Automated observations
    # Surface Airways observations (legacy hourly)
    "SAO",         # Airways including record specials
    "SAO-Airway",
    "SAOSP",       # Airways special excluding record specials
    "SA-AU",       # Airways auto merged
    "S-S-A",       # Synoptic airways auto merged
    # ASOS/AWOS automated observations
    "SOD",         # ASOS/AWOS
    "SOM",         # ASOS/AWOS
    # Other hourly sources
    "SMARS",       # Supplemental airways station
    "WBO",
    "WNO",         # Washington Nav Obs
    # Country-specific hourly data
    "EC DATABAS",  # Environment Canada database
    "EnvCan",      # Environment Canada
    "AUST",        # Australia
    "BRAZ",        # Brazil
    "GREEN",       # Greenland
    "MEXIC",       # Mexico
    # US networks with hourly data
    "CRN05",       # CRN 5-minute (sub-hourly)
    "CRN15",       # CRN 15-minute (sub-hourly)
    "COOPD",       # US Cooperative network SOD
    "MESOH",       # Hydro MESONET
    "MESOS",       # MESONET
    "MESOW",       # Snow MESONET
    "SHEF",        # Standard Hydro Exchange Format
    "PCP15",       # US 15-min precip network
    "PCP60",       # US 60-min precip network
}

# Report types that are summary/aggregate records (exclude these)
SUMMARY_REPORT_TYPES = {
    "FM-12",       # FM12: SYNOP fixed land station (typically 3-hourly summaries)
    "FM12-SYNOP",
    "FM-13",       # FM13: SHIP sea station
    "FM-14",       # FM14: SYNOP MOBIL mobile land station
    "FM-18",       # FM18: BUOY
    "SY-AE",       # Synop and aero merged
    "SY-AU",       # Synop and auto merged
    "SY-MT",       # Synop and METAR merged
    "SY-SA",       # Synop and airways merged
    "AERO",        # Aerological
}

# Column suffixes for each variable (6 attributes per variable)
VARIABLE_SUFFIXES = [
    "",  # The value itself
    "_Quality_Code",
    "_Measurement_Code",
    "_Report_Type_Code",
    "_Source_Code",
    "_units",
]


def load_station_year(station_id: str, year: int) -> pl.DataFrame | None:
    """Load a single station's data for a specific year.

    Args:
        station_id: Station identifier (e.g., "USW00003046")
        year: Year of the data

    Returns:
        Polars DataFrame or None if file doesn't exist
    """
    path = get_station_year_path(station_id, year)
    if not path.exists():
        logger.debug(f"No data found for station {station_id} year {year}")
        return None

    try:
        return pl.read_parquet(path)
    except Exception as e:
        logger.warning(f"Error reading {path}: {e}")
        return None


def _normalize_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize DataFrame schema to handle inconsistencies across years.

    Some parquet files have inconsistent schemas across years:
    - Numeric columns may be string-typed in some years
    - Quality code columns may be Struct types instead of Int32
    - Value columns may be Struct types instead of simple types

    This function normalizes these to consistent types before concatenation.

    Args:
        df: DataFrame that may have inconsistent column types

    Returns:
        DataFrame with normalized column types
    """
    # Columns that should be numeric (Float32/Float64)
    numeric_value_columns = [
        "temperature", "dew_point_temperature", "station_level_pressure",
        "sea_level_pressure", "wind_direction", "wind_speed", "wind_gust",
        "relative_humidity", "wet_bulb_temperature", "visibility",
        "precipitation", "pressure_3hr_change", "altimeter", "snow_depth",
        "Latitude", "Longitude",
    ]

    # Quality code columns that should be Int32
    quality_code_columns = [
        f"{var}_Quality_Code" for var in [
            "temperature", "dew_point_temperature", "station_level_pressure",
            "sea_level_pressure", "wind_direction", "wind_speed", "wind_gust",
            "relative_humidity", "wet_bulb_temperature", "visibility",
            "precipitation", "pressure_3hr_change", "altimeter", "snow_depth",
        ]
    ]

    cast_exprs = []

    # Handle numeric columns that may be string-typed
    for col in numeric_value_columns:
        if col in df.columns:
            dtype = df[col].dtype
            if dtype == pl.Utf8:
                cast_exprs.append(
                    pl.col(col).cast(pl.Float64, strict=False).alias(col)
                )
            elif isinstance(dtype, pl.Struct):
                # Extract first numeric field from struct
                for field in dtype.fields:
                    if field.dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
                        cast_exprs.append(
                            pl.col(col).struct.field(field.name).cast(pl.Float64, strict=False).alias(col)
                        )
                        break

    # Handle Elevation which is sometimes string
    if "Elevation" in df.columns and df["Elevation"].dtype == pl.Utf8:
        cast_exprs.append(
            pl.col("Elevation").cast(pl.Float64, strict=False).alias("Elevation")
        )

    # Handle quality code columns that may be Struct types
    for col in quality_code_columns:
        if col in df.columns:
            dtype = df[col].dtype
            if isinstance(dtype, pl.Struct):
                # Extract first integer field from struct (usually 'member0')
                for field in dtype.fields:
                    if field.dtype in (pl.Int32, pl.Int64, pl.Int16, pl.Int8):
                        cast_exprs.append(
                            pl.col(col).struct.field(field.name).alias(col)
                        )
                        break
                else:
                    # No integer field found, try first field
                    if dtype.fields:
                        cast_exprs.append(
                            pl.col(col).struct.field(dtype.fields[0].name).alias(col)
                        )

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df


def load_station_all_years(
    station_id: str,
    years: list[int] | None = None,
) -> pl.DataFrame | None:
    """Load all available data for a station across multiple years.

    Args:
        station_id: Station identifier
        years: Optional list of years to load. If None, loads all available.

    Returns:
        Combined Polars DataFrame or None if no data found
    """
    if years is None:
        years = list_available_years(station_id)

    if not years:
        return None

    dfs = []
    for year in years:
        df = load_station_year(station_id, year)
        if df is not None:
            # Normalize schema to handle inconsistencies across years
            df = _normalize_schema(df)
            dfs.append(df)

    if not dfs:
        return None

    # Concatenate all years
    # Use diagonal_relaxed to handle schema changes across years
    # (e.g., 2025+ added new columns like Year, Month, Day, Hour, Minute)
    return pl.concat(dfs, how="diagonal_relaxed")


def list_available_years(station_id: str) -> list[int]:
    """List all years with data for a given station.

    Args:
        station_id: Station identifier (e.g., "USW00003046")

    Returns:
        Sorted list of years with available data
    """
    years: list[int] = []

    if not GHCNH_RAW_DIR.exists():
        return years

    for year_dir in GHCNH_RAW_DIR.iterdir():
        if not year_dir.is_dir():
            continue

        try:
            year = int(year_dir.name)
        except ValueError:
            continue

        # File naming convention: GHCNh_{station_id}_{year}.parquet
        station_file = year_dir / f"GHCNh_{station_id}_{year}.parquet"
        if station_file.exists():
            years.append(year)

    return sorted(years)


def list_all_stations() -> list[str]:
    """List all unique station IDs across all years.

    Returns:
        Sorted list of unique station IDs (e.g., "USW00003046")
    """
    import re

    stations: set[str] = set()

    if not GHCNH_RAW_DIR.exists():
        return []

    # File naming convention: GHCNh_{station_id}_{year}.parquet
    pattern = re.compile(r"^GHCNh_(.+)_(\d{4})\.parquet$")

    for year_dir in GHCNH_RAW_DIR.iterdir():
        if not year_dir.is_dir():
            continue

        for parquet_file in year_dir.glob("*.parquet"):
            match = pattern.match(parquet_file.name)
            if match:
                station_id = match.group(1)
                stations.add(station_id)

    return sorted(stations)


def get_station_list(
    station_list_path: Path | None = None,
) -> pl.DataFrame | None:
    """Load the GHCNh station list metadata.

    The station list contains information about each station including:
    - Station ID
    - Latitude, Longitude, Elevation
    - State (for US stations)
    - Station name
    - GSN flag (GCOS Surface Network)
    - HCN/CRN flag (Historical Climatology / Climate Reference Network)
    - WMO ID

    Args:
        station_list_path: Path to station list CSV. Defaults to standard location.

    Returns:
        Polars DataFrame with station metadata, or None if not found
    """
    path = station_list_path or STATION_LIST_PATH

    if not path.exists():
        logger.warning(f"Station list not found at {path}")
        return None

    try:
        # Station list CSV has some fields that can be empty or contain text
        # Force string types for fields that might have mixed content
        return pl.read_csv(
            path,
            has_header=True,
            infer_schema_length=10000,
            schema_overrides={
                "WMO_ID": pl.Utf8,
                "ICAO": pl.Utf8,
                "GSN": pl.Utf8,
                "(US)HCN_(US)CRN": pl.Utf8,
                "STATE": pl.Utf8,
            },
            null_values=["", "NA", "N/A"],
        )
    except Exception as e:
        logger.warning(f"Error reading station list: {e}")
        return None


def get_available_columns(station_id: str, year: int) -> list[str]:
    """Get list of columns available in a station's data file.

    Args:
        station_id: Station identifier
        year: Year of the data

    Returns:
        List of column names
    """
    path = get_station_year_path(station_id, year)
    if not path.exists():
        return []

    try:
        schema = pl.read_parquet_schema(path)
        return list(schema.keys())
    except Exception as e:
        logger.warning(f"Error reading schema for {path}: {e}")
        return []


def filter_hourly_records(
    df: pl.DataFrame,
    report_type_column: str = "temperature_Report_Type",
) -> pl.DataFrame:
    """Filter DataFrame to include only hourly instrument records.

    Excludes summary/aggregate record types like FM-12 SYNOP (3-hourly summaries).
    Records with unknown report types or null report types are included by default
    (conservative approach - don't exclude data we're uncertain about).

    Args:
        df: DataFrame with weather data
        report_type_column: Column containing report type codes

    Returns:
        Filtered DataFrame with only hourly instrument records
    """
    if report_type_column not in df.columns:
        # If no report type column, return all data
        return df

    # Build exclusion filter for known summary types
    # We exclude records that explicitly match summary types
    # Records with null/unknown types are kept (conservative)
    exclusion_mask = pl.lit(False)
    for summary_type in SUMMARY_REPORT_TYPES:
        exclusion_mask = exclusion_mask | (pl.col(report_type_column) == summary_type)
        # Also check for partial matches (e.g., "FM12-SYNOP" contains "FM12")
        exclusion_mask = exclusion_mask | pl.col(report_type_column).str.starts_with(summary_type)

    return df.filter(~exclusion_mask)


def count_report_types(
    df: pl.DataFrame,
    report_type_column: str = "temperature_Report_Type",
) -> dict[str, int]:
    """Count occurrences of each report type in the DataFrame.

    Args:
        df: DataFrame with weather data
        report_type_column: Column containing report type codes

    Returns:
        Dictionary mapping report type to count
    """
    if report_type_column not in df.columns:
        return {}

    counts = df.group_by(report_type_column).agg(pl.count().alias("count"))
    return {
        str(row[report_type_column]) if row[report_type_column] is not None else "null": row["count"]
        for row in counts.iter_rows(named=True)
    }


def is_hourly_report_type(report_type: str | None) -> bool:
    """Check if a report type represents hourly instrument data.

    Args:
        report_type: Report type code

    Returns:
        True if it's a known hourly type or unknown (conservative),
        False if it's a known summary/aggregate type
    """
    if report_type is None:
        return True  # Conservative: include unknown types

    # Check against summary types
    for summary_type in SUMMARY_REPORT_TYPES:
        if report_type == summary_type or report_type.startswith(summary_type):
            return False

    return True


def extract_tier1_variables(
    df: pl.DataFrame,
    variables: list[str] | None = None,
) -> pl.DataFrame:
    """Extract Tier 1 weather variables with all their attributes.

    Extracts the specified weather variables (or all Tier 1 variables if not specified)
    along with their associated attributes (value, Quality_Code, Measurement_Code,
    Report_Type_Code, Source_Code, units). Also includes primary columns
    (STATION, Station_name, DATE, LATITUDE, LONGITUDE).

    Args:
        df: DataFrame containing GHCNh parquet data
        variables: List of variable names to extract. If None, extracts all Tier 1 variables.

    Returns:
        DataFrame containing only primary columns and requested variables with their attributes.

    Example:
        >>> df = load_station_year("USW00003046", 2023)
        >>> tier1_df = extract_tier1_variables(df)
        >>> # Returns DataFrame with PRIMARY_COLUMNS + all TIER1_VARIABLES + their 6 attributes each
    """
    if variables is None:
        variables = TIER1_VARIABLES

    # Start with primary columns
    columns_to_select = list(PRIMARY_COLUMNS)

    # Add each variable and all its attributes
    for var in variables:
        for suffix in VARIABLE_SUFFIXES:
            col_name = f"{var}{suffix}" if suffix else var
            # Only include columns that exist in the DataFrame
            if col_name in df.columns:
                columns_to_select.append(col_name)

    # Select only the columns that exist
    available_columns = [col for col in columns_to_select if col in df.columns]

    if not available_columns:
        logger.warning("No requested columns found in DataFrame")
        return pl.DataFrame()

    return df.select(available_columns)


def filter_by_quality_flags(
    df: pl.DataFrame,
    variables: list[str] | None = None,
    exclude_erroneous: bool = True,
    exclude_suspect: bool = False,
) -> pl.DataFrame:
    """Filter observations by quality control flags.

    Removes observations with poor quality flags based on GHCNh quality code definitions.
    By default, excludes erroneous values (QC codes indicating data errors).

    Quality code definitions (from GHCNh documentation Section VI):
    - Legacy codes for sources 313-346:
      - 0: Passed gross limits check
      - 1: Passed all QC checks
      - 2: Suspect
      - 3: Erroneous
      - 4-9, A-Z: Various QC flags
    - Legacy codes for sources 220-223, 347-348:
      - 0: Not checked
      - 1: Good
      - 2: Suspect
      - 3: Erroneous

    Args:
        df: DataFrame containing GHCNh data with Quality_Code columns
        variables: List of variables to filter. If None, filters all Tier 1 variables.
        exclude_erroneous: If True, set erroneous values to null (default: True)
        exclude_suspect: If True, also set suspect values to null (default: False)

    Returns:
        DataFrame with poor-quality observations set to null for each variable.
        Quality_Code columns are preserved for reference.

    Example:
        >>> df = load_station_year("USW00003046", 2023)
        >>> df_filtered = filter_by_quality_flags(df)
        >>> # Erroneous temperature values are now null
    """
    if variables is None:
        variables = TIER1_VARIABLES

    # Build list of quality codes to exclude
    exclude_codes = []
    if exclude_erroneous:
        exclude_codes.extend(["3", "7"])  # Erroneous codes
    if exclude_suspect:
        exclude_codes.extend(["2", "6"])  # Suspect codes

    if not exclude_codes:
        logger.info("No quality codes to filter - returning original DataFrame")
        return df

    # Apply quality filtering for each variable
    filter_expressions = []
    for var in variables:
        qc_col = f"{var}_Quality_Code"

        if qc_col not in df.columns or var not in df.columns:
            continue

        # Set variable value to null when quality code indicates bad data
        # Use when().then().otherwise() to conditionally set values
        filtered_value = pl.when(
            pl.col(qc_col).cast(pl.Utf8).is_in(exclude_codes)
        ).then(None).otherwise(pl.col(var))

        filter_expressions.append(filtered_value.alias(var))

    if not filter_expressions:
        logger.warning("No quality code columns found - returning original DataFrame")
        return df

    # Apply all filter expressions at once
    return df.with_columns(filter_expressions)
