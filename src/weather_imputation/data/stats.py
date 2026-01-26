"""Statistics calculations for weather data."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl

logger = logging.getLogger(__name__)

# Quality codes that indicate valid data
# Based on GHCNh documentation
# Note: Quality codes can be either integers or strings depending on file format
VALID_QUALITY_CODES_INT = {1, 5}  # Integer quality codes: 1=passed, 5=NCEI passed
VALID_QUALITY_CODES_STR = {"", "1", "5", "A", "C"}  # String quality codes

# Quality codes that indicate invalid/suspect data
INVALID_QUALITY_CODES_INT = {3, 7}  # Integer: 3=failed climatological, 7=failed QC
INVALID_QUALITY_CODES_STR = {
    "3",  # Failed climatological check
    "7",  # Failed QC check
    "L",  # Suspiciously low
    "o",  # Suspiciously high (old format)
    "F",  # Failed QC
    "I",  # Inconsistent
    "M",  # Manual bad
    "N",  # Not representative
    "R",  # Replaced
    "S",  # Suspect
    "W",  # Wrong units
    "X",  # Fail
    "Z",  # Zero suspect
}


@dataclass
class VariableStats:
    """Statistics for a single weather variable."""

    count: int
    valid_count: int
    null_count: int
    completeness_pct: float
    mean: float | None
    std: float | None
    min_val: float | None
    max_val: float | None


@dataclass
class GapAnalysis:
    """Analysis of gaps in time series data."""

    gap_count_24h: int  # Number of gaps > 24 hours
    max_gap_hours: float
    avg_observation_interval_hours: float
    total_observations: int
    expected_observations: int  # Based on hourly frequency


def _normalize_quality_column(df: pl.DataFrame, quality_column: str) -> pl.Series:
    """Normalize a quality code column to a usable format.

    Some parquet files have quality code columns as Struct types (e.g.,
    Struct({'member0': Int32, 'member1': String})) instead of simple Int32.
    This function extracts the integer component from Struct columns.

    Args:
        df: DataFrame with weather data
        quality_column: Name of the quality code column

    Returns:
        Normalized quality code Series (Int32 or Utf8)
    """
    qc_col = df[quality_column]
    dtype = qc_col.dtype

    # Handle Struct type - extract the first integer field
    if isinstance(dtype, pl.Struct):
        # Get field names from the struct
        field_names = [field.name for field in dtype.fields]
        # Try to find an integer field (usually 'member0' contains the QC code)
        for field in dtype.fields:
            if field.dtype in (pl.Int32, pl.Int64, pl.Int16, pl.Int8):
                return qc_col.struct.field(field.name)
        # If no integer field, try first field
        if field_names:
            return qc_col.struct.field(field_names[0])

    return qc_col


def _get_valid_quality_codes(dtype: pl.DataType) -> list[int | str]:
    """Get the appropriate valid quality codes list based on column dtype."""
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        return list(VALID_QUALITY_CODES_INT)
    else:
        return list(VALID_QUALITY_CODES_STR)


def _get_invalid_quality_codes(dtype: pl.DataType) -> list[int | str]:
    """Get the appropriate invalid quality codes list based on column dtype."""
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        return list(INVALID_QUALITY_CODES_INT)
    else:
        return list(INVALID_QUALITY_CODES_STR)


def compute_completeness(
    df: pl.DataFrame,
    column: str,
    quality_column: str | None = None,
) -> float:
    """Compute completeness percentage for a column.

    Completeness is the percentage of non-null values that pass quality checks.

    Args:
        df: DataFrame with weather data
        column: Name of the value column
        quality_column: Name of the quality code column (optional)

    Returns:
        Completeness as a percentage (0-100)
    """
    if column not in df.columns:
        return 0.0

    total_rows = len(df)
    if total_rows == 0:
        return 0.0

    # Start with non-null values
    valid_mask = df[column].is_not_null()

    # Apply quality filter if quality column exists
    if quality_column and quality_column in df.columns:
        qc_col = _normalize_quality_column(df, quality_column)
        # Get appropriate valid codes based on column dtype
        valid_codes = _get_valid_quality_codes(qc_col.dtype)
        # Consider null quality codes as valid (not checked)
        quality_ok = qc_col.is_null() | qc_col.is_in(valid_codes)
        valid_mask = valid_mask & quality_ok

    valid_count = valid_mask.sum()
    return (valid_count / total_rows) * 100


def compute_variable_stats(
    df: pl.DataFrame,
    column: str,
    quality_column: str | None = None,
) -> VariableStats:
    """Compute statistics for a weather variable.

    Args:
        df: DataFrame with weather data
        column: Name of the value column
        quality_column: Name of the quality code column (optional)

    Returns:
        VariableStats dataclass with computed statistics
    """
    if column not in df.columns:
        return VariableStats(
            count=0,
            valid_count=0,
            null_count=len(df),
            completeness_pct=0.0,
            mean=None,
            std=None,
            min_val=None,
            max_val=None,
        )

    total = len(df)
    null_count = df[column].null_count()

    # Filter for valid data
    valid_mask = df[column].is_not_null()

    if quality_column and quality_column in df.columns:
        qc_col = _normalize_quality_column(df, quality_column)
        valid_codes = _get_valid_quality_codes(qc_col.dtype)
        quality_ok = qc_col.is_null() | qc_col.is_in(valid_codes)
        valid_mask = valid_mask & quality_ok

    valid_df = df.filter(valid_mask)
    valid_count = len(valid_df)
    completeness = (valid_count / total * 100) if total > 0 else 0.0

    # Compute statistics on valid data
    if valid_count > 0:
        # Ensure column is numeric before computing stats
        col_dtype = valid_df[column].dtype
        if col_dtype == pl.Utf8:
            # Try to cast string column to float
            try:
                valid_df = valid_df.with_columns(
                    pl.col(column).cast(pl.Float64, strict=False).alias(column)
                )
            except Exception:
                # If cast fails, return None for stats
                return VariableStats(
                    count=total,
                    valid_count=valid_count,
                    null_count=null_count,
                    completeness_pct=completeness,
                    mean=None,
                    std=None,
                    min_val=None,
                    max_val=None,
                )

        try:
            stats_df = valid_df.select([
                pl.col(column).mean().alias("mean"),
                pl.col(column).std().alias("std"),
                pl.col(column).min().alias("min"),
                pl.col(column).max().alias("max"),
            ])
            row = stats_df.row(0)
            mean_val, std_val, min_val, max_val = row
        except Exception:
            # If stats computation fails (e.g., incompatible dtype), return None
            mean_val = std_val = min_val = max_val = None
    else:
        mean_val = std_val = min_val = max_val = None

    return VariableStats(
        count=total,
        valid_count=valid_count,
        null_count=null_count,
        completeness_pct=completeness,
        mean=mean_val,
        std=std_val,
        min_val=min_val,
        max_val=max_val,
    )


def compute_gap_analysis(df: pl.DataFrame, date_column: str = "DATE") -> GapAnalysis:
    """Analyze gaps in the time series data.

    Args:
        df: DataFrame with weather data
        date_column: Name of the datetime column

    Returns:
        GapAnalysis dataclass with gap statistics
    """
    if date_column not in df.columns or len(df) < 2:
        return GapAnalysis(
            gap_count_24h=0,
            max_gap_hours=0.0,
            avg_observation_interval_hours=0.0,
            total_observations=len(df),
            expected_observations=0,
        )

    # Sort by date and compute time differences
    sorted_df = df.sort(date_column)
    dates = sorted_df[date_column]

    # Calculate differences between consecutive observations
    diff_df = sorted_df.with_columns([
        (pl.col(date_column).diff().dt.total_seconds() / 3600).alias("gap_hours")
    ]).filter(pl.col("gap_hours").is_not_null())

    if len(diff_df) == 0:
        return GapAnalysis(
            gap_count_24h=0,
            max_gap_hours=0.0,
            avg_observation_interval_hours=0.0,
            total_observations=len(df),
            expected_observations=0,
        )

    # Count gaps > 24 hours
    gap_count_24h = diff_df.filter(pl.col("gap_hours") > 24).height

    # Max gap
    max_gap = diff_df["gap_hours"].max() or 0.0

    # Average interval
    avg_interval = diff_df["gap_hours"].mean() or 0.0

    # Calculate expected observations (hourly data)
    first_date = dates.min()
    last_date = dates.max()
    if first_date is not None and last_date is not None:
        total_hours = (last_date - first_date).total_seconds() / 3600
        expected_obs = int(total_hours) + 1  # Include both endpoints
    else:
        expected_obs = 0

    return GapAnalysis(
        gap_count_24h=gap_count_24h,
        max_gap_hours=float(max_gap),
        avg_observation_interval_hours=float(avg_interval),
        total_observations=len(df),
        expected_observations=expected_obs,
    )


def filter_by_quality_code(
    df: pl.DataFrame,
    column: str,
    quality_column: str,
    keep_valid: bool = True,
) -> pl.DataFrame:
    """Filter DataFrame by quality code.

    Args:
        df: DataFrame with weather data
        column: Name of the value column
        quality_column: Name of the quality code column
        keep_valid: If True, keep valid data; if False, keep invalid data

    Returns:
        Filtered DataFrame
    """
    if quality_column not in df.columns:
        return df if keep_valid else df.filter(pl.lit(False))

    qc_col = _normalize_quality_column(df, quality_column)

    if keep_valid:
        # Keep null quality codes and valid codes
        valid_codes = _get_valid_quality_codes(qc_col.dtype)
        mask = qc_col.is_null() | qc_col.is_in(valid_codes)
    else:
        # Keep only invalid codes
        invalid_codes = _get_invalid_quality_codes(qc_col.dtype)
        mask = qc_col.is_in(invalid_codes)

    return df.filter(mask)
