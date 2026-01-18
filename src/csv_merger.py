#!/usr/bin/env python3
"""
Functions for merging and processing CSV data, with Polars integration for performance.
"""
import logging
import csv
import io
from io import StringIO
from typing import List, Dict, Tuple
import polars as pl

from config import METADATA_FIELDS, METADATA_ONLY_FIELDS
from utils import safe_csv_read

logger = logging.getLogger("weather_processor")


def merge_headers(existing_header: List[str], new_header: List[str]) -> List[str]:
    """
    Merge two different headers to accommodate all fields.
    """
    # Start with existing header
    merged = existing_header.copy()

    # Add any fields from new header that don't exist in merged
    for field in new_header:
        if field not in merged:
            merged.append(field)

    return merged


def get_canonical_headers(headers: List[str]) -> List[str]:
    """
    Order headers based on ISD format specification.

    Ensures consistent column ordering following the ISD specification:
    1. Metadata columns: STATION, DATE, SOURCE, REPORT_TYPE, CALL_SIGN, QUALITY_CONTROL
    2. Mandatory data columns: WND, CIG, VIS, TMP, DEW, SLP
    3. Optional data columns (alphabetically sorted) - preserving numeric suffixes
    4. Final columns: REM, EQD

    Args:
        headers: List of column headers to order

    Returns:
        List of headers in canonical order
    """
    # Define the metadata columns in their required order
    metadata_cols = [
        "STATION",
        "DATE",
        "SOURCE",
        "REPORT_TYPE",
        "CALL_SIGN",
        "QUALITY_CONTROL",
    ]

    # Define the mandatory data columns in their required order
    mandatory_cols = ["WND", "CIG", "VIS", "TMP", "DEW", "SLP"]

    # Define the final columns in their required order
    final_cols = ["REM", "EQD"]

    # Initialize canonical headers with metadata columns that exist
    canonical_headers = [col for col in metadata_cols if col in headers]

    # Add mandatory columns that exist
    canonical_headers.extend([col for col in mandatory_cols if col in headers])

    # Collect optional columns (excluding metadata, mandatory, and final columns)
    optional_cols = [
        col
        for col in headers
        if col not in metadata_cols
        and col not in mandatory_cols
        and col not in final_cols
    ]

    # Sort optional columns alphabetically and add them to canonical headers
    canonical_headers.extend(sorted(optional_cols))

    # Add final columns that exist
    canonical_headers.extend([col for col in final_cols if col in headers])

    return canonical_headers


def filter_csv_data(content: str) -> Tuple[str, Dict[str, str]]:
    """
    Filter CSV data to separate metadata and weather observations.

    Returns:
        Tuple of (filtered_content, metadata)
        - filtered_content: CSV with only weather data (no metadata)
        - metadata: Dictionary with station metadata
    """

    # Parse CSV content safely
    header, data_rows = safe_csv_read(content)
    if not header or not data_rows:
        return "", {}

    # Extract metadata from first row
    metadata = {}
    if data_rows:
        first_row = data_rows[0]
        for field in METADATA_FIELDS:
            try:
                idx = header.index(field)
                if idx < len(first_row):
                    metadata[field] = first_row[idx]
            except (ValueError, IndexError):
                metadata[field] = ""

    # Determine which columns to exclude (metadata only)
    indices_to_exclude = []
    column_names = []

    for i, col in enumerate(header):
        if col in METADATA_ONLY_FIELDS:
            indices_to_exclude.append(i)
        else:
            column_names.append(col)

    # Create a new CSV with only the weather observation data
    output = io.StringIO()
    writer = csv.writer(output, quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Add header without metadata fields
    writer.writerow(column_names)

    # Process data rows
    for row in data_rows:
        filtered_row = [row[i] for i in range(len(row)) if i not in indices_to_exclude]
        writer.writerow(filtered_row)

    return output.getvalue(), metadata


def merge_csv_data(existing_content: str, new_content: str) -> str:
    """
    Fallback merge function for CSV content when Polars merge fails.
    Handles quoted fields with commas correctly.
    """
    # Parse existing content
    existing_reader = csv.reader(
        StringIO(existing_content), quotechar='"', quoting=csv.QUOTE_MINIMAL
    )
    existing_rows = list(existing_reader)

    if not existing_rows:
        return new_content

    existing_header = existing_rows[0]
    existing_data = existing_rows[1:] if len(existing_rows) > 1 else []

    # Parse new content
    new_reader = csv.reader(
        StringIO(new_content), quotechar='"', quoting=csv.QUOTE_MINIMAL
    )
    new_rows = list(new_reader)

    if not new_rows:
        return existing_content

    new_header = new_rows[0]
    new_data = new_rows[1:] if len(new_rows) > 1 else []

    # Check if headers match
    if existing_header == new_header:
        # Headers match, just append data rows
        output = io.StringIO()
        writer = csv.writer(output, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(existing_header)
        writer.writerows(existing_data)
        writer.writerows(new_data)
        return output.getvalue()

    # Combine all unique headers
    all_headers = list(
        dict.fromkeys(
            existing_header + [col for col in new_header if col not in existing_header]
        )
    )

    # Get canonical ordering
    canonical_headers = get_canonical_headers(all_headers)

    # Create index mappings for the original headers
    existing_indices = {col: i for i, col in enumerate(existing_header)}
    new_indices = {col: i for i, col in enumerate(new_header)}

    # Function to map a row to the canonical headers
    def map_row_to_canonical(row, indices):
        new_row = [""] * len(canonical_headers)
        for i, col in enumerate(canonical_headers):
            if col in indices and indices[col] < len(row):
                new_row[i] = row[indices[col]]
        return new_row

    # Map all rows to the canonical headers
    merged_data = []
    for row in existing_data:
        merged_data.append(map_row_to_canonical(row, existing_indices))
    for row in new_data:
        merged_data.append(map_row_to_canonical(row, new_indices))

    # Write the merged data
    output = io.StringIO()
    writer = csv.writer(output, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(canonical_headers)
    writer.writerows(merged_data)

    return output.getvalue()


def merge_csv_data_with_polars(existing_content: str, new_content: str) -> str:
    """
    Merge CSV content using Polars for better performance.
    Optimized to reduce string conversions and use efficient Polars operations.
    Handles quoted fields with internal commas correctly.
    """

    # Parse CSV content using Polars' native CSV reader
    def parse_csv_for_polars(content, source_name):
        try:
            if not content.strip():
                logger.warning(f"Empty content in {source_name}")
                return None

            # Use Polars' native CSV parser with proper settings
            # Read directly from StringIO without intermediate conversions
            df = pl.read_csv(
                StringIO(content),
                has_header=True,
                quote_char='"',
                ignore_errors=True,
                infer_schema_length=0,  # Don't infer schema types
                try_parse_dates=False,  # We'll handle dates explicitly
            )

            if df.shape[0] == 0:
                logger.warning(f"No data rows found in {source_name}")
                return None

            # Handle DATE column parsing if present
            # Try with microseconds first, then without (NOAA data has inconsistent formats)
            if "DATE" in df.columns:
                df = df.with_columns(
                    [pl.col("DATE").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False)
                     .fill_null(pl.col("DATE").str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False))]
                )

            # Fill nulls and cast non-date columns to strings in one operation
            non_date_cols = [col for col in df.columns if col != "DATE"]
            if non_date_cols:
                df = df.with_columns(
                    [pl.col(non_date_cols).fill_null("").cast(pl.Utf8)]
                )

            return df

        except Exception as e:
            logger.error(f"Error parsing {source_name} with Polars: {e}")
            return None

    try:
        # Parse CSV files with proper quote handling
        existing_df = parse_csv_for_polars(existing_content, "existing file")

        if existing_df is None:
            logger.warning(
                "Failed to parse existing file, falling back to standard merge"
            )
            return merge_csv_data(existing_content, new_content)

        new_df = parse_csv_for_polars(new_content, "new file")

        if new_df is None:
            logger.warning("Failed to parse new file, falling back to standard merge")
            return merge_csv_data(existing_content, new_content)

        # Merge and process dataframes in one optimized chain
        # Use diagonal concat to handle schema differences, then deduplicate
        merged_df = (
            pl.concat([existing_df, new_df], how="diagonal")
            .unique()
            .fill_null("")
        )

        # Get canonical headers and reorder columns
        canonical_headers = get_canonical_headers(merged_df.columns)
        merged_df = merged_df.select([pl.col(col) for col in canonical_headers])

        # Format DATE column back to string with ISO format before writing
        if "DATE" in merged_df.columns and merged_df["DATE"].dtype == pl.Datetime:
            merged_df = merged_df.with_columns(
                [pl.col("DATE").dt.strftime("%Y-%m-%dT%H:%M:%S")]
            )

        # Cast all columns to strings to ensure consistent output
        merged_df = merged_df.select(
            [pl.col(col).cast(pl.Utf8) for col in merged_df.columns]
        )

        # Convert back to CSV with proper quoting - write to StringIO
        output = io.StringIO()
        merged_df.write_csv(output, quote_style="necessary")

        return output.getvalue()

    except Exception as e:
        logger.error(f"Error in Polars merge: {e}", exc_info=True)
        logger.warning("Falling back to standard CSV merge")
        return merge_csv_data(existing_content, new_content)
