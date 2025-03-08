#!/usr/bin/env python3
"""
Functions for merging and processing CSV data, with Polars integration for performance.
"""
import time
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

def get_canonical_headers(header1: List[str], header2: List[str]) -> List[str]:
    """
    Generate canonical headers based on ISD format specification.
    
    Ensures consistent column ordering following the ISD specification:
    1. Metadata columns: STATION, DATE, SOURCE, REPORT_TYPE, CALL_SIGN, QUALITY_CONTROL
    2. Mandatory data columns: WND, CIG, VIS, TMP, DEW, SLP
    3. Optional data columns (preserving original order)
    4. Final columns: REM, EQD
    
    Args:
        header1: First header list
        header2: Second header list
    
    Returns:
        List of headers in canonical order
    """
    # Create a set of all unique column names from both headers
    all_columns = set(header1).union(set(header2))
    
    # Define the metadata columns in their required order
    metadata_cols = ["STATION", "DATE", "SOURCE", "REPORT_TYPE", "CALL_SIGN", "QUALITY_CONTROL"]
    
    # Define the mandatory data columns in their required order
    mandatory_cols = ["WND", "CIG", "VIS", "TMP", "DEW", "SLP"]
    
    # Define the final columns in their required order
    final_cols = ["REM", "EQD"]
    
    # Initialize canonical headers with metadata and mandatory columns that exist
    canonical_headers = [col for col in metadata_cols if col in all_columns]
    canonical_headers.extend([col for col in mandatory_cols if col in all_columns])
    
    # Collect optional columns (excluding metadata, mandatory, and final columns)
    optional_cols = all_columns - set(metadata_cols) - set(mandatory_cols) - set(final_cols)
    
    # Get the column prefixes (e.g., "AH" from "AH1")
    column_prefixes = {}
    for col in optional_cols:
        # Extract letter prefix and digits
        prefix = ''.join([c for c in col if c.isalpha()])
        if prefix:
            if prefix not in column_prefixes:
                column_prefixes[prefix] = []
            column_prefixes[prefix].append(col)
    
    # Add normalized optional columns to the canonical headers
    for prefix in sorted(column_prefixes.keys()):
        # If there's only one column with this prefix, use the original name
        if len(column_prefixes[prefix]) == 1:
            canonical_headers.append(column_prefixes[prefix][0])
        else:
            # Otherwise, standardize to just the prefix (like "AH" for "AH1", "AH2", etc.)
            canonical_headers.append(prefix)
    
    # Add final columns that exist
    canonical_headers.extend([col for col in final_cols if col in all_columns])
    
    logger.debug(f"Generated canonical headers: {canonical_headers}")
    
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
    Merge new CSV content with existing content, handling potential header differences.
    Properly handles quoted fields with commas.
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

    logger.info(f"Merging headers: \n old: {existing_header} \n new: {new_header}")

    # Generate canonical headers
    canonical_header = get_canonical_headers(existing_header, new_header)
    
    # Create mapping from original columns to canonical columns
    existing_map = {}
    for col in existing_header:
        prefix = ''.join([c for c in col if c.isalpha()])
        if prefix and prefix in canonical_header and prefix != col:
            existing_map[col] = prefix
        else:
            existing_map[col] = col
    
    new_map = {}
    for col in new_header:
        prefix = ''.join([c for c in col if c.isalpha()])
        if prefix and prefix in canonical_header and prefix != col:
            new_map[col] = prefix
        else:
            new_map[col] = col
    
    # Create reverse mappings for data access
    existing_index_map = {existing_map.get(col, col): i for i, col in enumerate(existing_header)}
    new_index_map = {new_map.get(col, col): i for i, col in enumerate(new_header)}

    # Map existing data to canonical headers
    merged_data = []
    for row in existing_data:
        new_row = [""] * len(canonical_header)
        for i, col in enumerate(canonical_header):
            if col in existing_index_map and existing_index_map[col] < len(row):
                new_row[i] = row[existing_index_map[col]]
        merged_data.append(new_row)

    # Map new data to canonical headers
    for row in new_data:
        new_row = [""] * len(canonical_header)
        for i, col in enumerate(canonical_header):
            if col in new_index_map and new_index_map[col] < len(row):
                new_row[i] = row[new_index_map[col]]
        merged_data.append(new_row)

    # Write the merged data
    output = io.StringIO()
    writer = csv.writer(output, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(canonical_header)
    writer.writerows(merged_data)

    return output.getvalue()


def merge_csv_data_with_polars(existing_content: str, new_content: str) -> str:
    """
    Merge CSV content using Polars for better performance.
    Handles quoted fields with internal commas correctly.
    Includes detailed logging for performance monitoring.
    """
    logger.debug("Starting Polars CSV merge operation")
    start_time = time.time()

    # Parse CSV content with proper quote handling first
    def parse_csv_for_polars(content, source_name):

        try:
            # Use Python's CSV reader with proper quote handling
            csv_reader = csv.reader(
                StringIO(content), quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            rows = list(csv_reader)

            if not rows:
                logger.warning(f"No data found in {source_name}")
                return None

            columns = rows[0]
            data_rows = rows[1:]

            if not data_rows:
                logger.warning(f"No data rows found in {source_name}")
                return pl.DataFrame(schema=[pl.Field(col, pl.Utf8) for col in columns])

            # Check if any rows have more fields than headers (only log once)
            max_fields = max(len(row) for row in data_rows)
            if max_fields > len(columns):
                # Log just once with a summary instead of for every row
                logger.info(
                    f"Some rows in {source_name} have more fields than the header ({max_fields} > {len(columns)}). "
                )

            # Create a dictionary of columns for Polars
            data_dict = {col: [] for col in columns}

            for row in data_rows:
                for i, col in enumerate(columns):
                    if i < len(row):
                        data_dict[col].append(row[i])
                    else:
                        data_dict[col].append("")

            # Create DataFrame with explicit string type
            return pl.DataFrame(data_dict).select(
                [pl.col(col).cast(pl.Utf8) for col in columns]
            )

        except Exception as e:
            logger.error(f"Error parsing {source_name}: {e}")
            return None

    try:
        # Parse CSV files with proper quote handling
        parse_start = time.time()
        existing_df = parse_csv_for_polars(existing_content, "existing file")
        parse_time = time.time() - parse_start
        logger.debug(
            f"Parsed existing content ({len(existing_content)} bytes) in {parse_time:.2f}s"
        )

        if existing_df is None:
            logger.warning(
                "Failed to parse existing file, falling back to standard merge"
            )
            return merge_csv_data(existing_content, new_content)

        parse_start = time.time()
        new_df = parse_csv_for_polars(new_content, "new file")
        parse_time = time.time() - parse_start
        logger.debug(
            f"Parsed new content ({len(new_content)} bytes) in {parse_time:.2f}s"
        )

        if new_df is None:
            logger.warning("Failed to parse new file, falling back to standard merge")
            return merge_csv_data(existing_content, new_content)

        # Get column information for both dataframes
        existing_cols = existing_df.columns
        new_cols = new_df.columns

        # Process and normalize column headers
        canonical_headers = get_canonical_headers(existing_cols, new_cols)
        
        # Create column mapping for existing dataframe
        existing_col_map = {}
        for col in existing_cols:
            prefix = ''.join([c for c in col if c.isalpha()])
            if prefix and prefix in canonical_headers and prefix != col:
                # Map from original column to standardized prefix
                existing_col_map[col] = prefix
        
        # Create column mapping for new dataframe
        new_col_map = {}
        for col in new_cols:
            prefix = ''.join([c for c in col if c.isalpha()])
            if prefix and prefix in canonical_headers and prefix != col:
                # Map from original column to standardized prefix
                new_col_map[col] = prefix
        
        # Rename columns in existing dataframe
        if existing_col_map:
            existing_df = existing_df.rename(existing_col_map)
        
        # Rename columns in new dataframe
        if new_col_map:
            new_df = new_df.rename(new_col_map)
        
        # Add missing columns to each dataframe
        for col in canonical_headers:
            if col not in existing_df.columns:
                existing_df = existing_df.with_columns(pl.lit("").cast(pl.Utf8).alias(col))
            if col not in new_df.columns:
                new_df = new_df.with_columns(pl.lit("").cast(pl.Utf8).alias(col))
        
        # Ensure both dataframes have the same column ordering
        existing_df = existing_df.select([pl.col(col).cast(pl.Utf8) for col in canonical_headers])
        new_df = new_df.select([pl.col(col).cast(pl.Utf8) for col in canonical_headers])

        # Concatenate dataframes
        merge_start = time.time()
        merged_df = pl.concat([existing_df, new_df])

        # Deduplicate if needed
        if merged_df.shape[0] > existing_df.shape[0] + new_df.shape[0]:
            logger.debug("Potential duplicates detected, performing deduplication")
            merged_df = merged_df.unique()

        merge_time = time.time() - merge_start
        logger.debug(f"Merged dataframes in {merge_time:.2f}s")

        # Convert back to CSV with proper quoting
        csv_start = time.time()
        output = io.StringIO()
        merged_df.write_csv(output, quote_style="necessary")
        csv_time = time.time() - csv_start
        logger.debug(f"Converted to CSV in {csv_time:.2f}s")

        total_time = time.time() - start_time
        logger.debug(f"Total Polars merge operation completed in {total_time:.2f}s")

        return output.getvalue()

    except Exception as e:
        logger.error(f"Error in Polars merge: {e}", exc_info=True)
        logger.warning("Falling back to standard CSV merge")
        return merge_csv_data(existing_content, new_content)
