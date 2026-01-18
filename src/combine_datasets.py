#!/usr/bin/env python3
"""
Script to combine two merged datasets (merged/ and merged-1/) into a single dataset.
Supports both CSV.gz and Parquet formats for optimal performance.

This script:
1. Identifies all stations from both directories
2. Merges overlapping stations chronologically
3. Deduplicates records
4. Merges metadata from both SQLite databases and CSV files
5. Combines year counts from both datasets
6. Outputs to a new directory (merged-combined/)
7. Optionally converts to Parquet format for better performance

Note: The two datasets are typically different year ranges:
- merged/: beginning through ~2007
- merged-1/: 2008-2024/25
"""
import argparse
import gzip
import logging
import sqlite3
from pathlib import Path
from typing import Set, Optional
from tqdm import tqdm
import polars as pl

from config import DATA_DIR

logger = logging.getLogger("weather_processor")


def get_all_station_ids(merged_dir: Path, merged1_dir: Path) -> Set[str]:
    """
    Get all unique station IDs from both directories.
    
    Args:
        merged_dir: Path to first merged directory
        merged1_dir: Path to second merged directory
        
    Returns:
        Set of all station IDs found in either directory
    """
    stations_merged = {f.stem.replace(".csv", "").replace(".parquet", "") 
                       for f in merged_dir.glob("*.csv.gz")} | \
                      {f.stem for f in merged_dir.glob("*.parquet")}
    
    stations_merged1 = {f.stem.replace(".csv", "").replace(".parquet", "") 
                        for f in merged1_dir.glob("*.csv.gz")} | \
                       {f.stem for f in merged1_dir.glob("*.parquet")}
    
    all_stations = stations_merged | stations_merged1
    logger.info(
        "Found %d stations in merged/, %d stations in merged-1/, %d unique stations total",
        len(stations_merged),
        len(stations_merged1),
        len(all_stations)
    )
    return all_stations


def read_station_file(file_path: Path) -> Optional[pl.DataFrame]:
    """
    Read a station file, supporting both CSV.gz and Parquet formats.
    
    Args:
        file_path: Path to the station file
        
    Returns:
        Polars DataFrame or None if read fails
    """
    try:
        if file_path.suffix == ".gz" or file_path.suffixes == [".csv", ".gz"]:
            # Read gzipped CSV
            with gzip.open(file_path, "rb") as f:
                df = pl.read_csv(
                    f,
                    has_header=True,
                    quote_char='"',
                    ignore_errors=True,
                    infer_schema_length=0,
                    try_parse_dates=False,
                )
        elif file_path.suffix == ".parquet":
            # Read Parquet
            df = pl.read_parquet(file_path)
        else:
            logger.warning("Unknown file format for %s", file_path)
            return None
        
        # Handle DATE column if present
        if "DATE" in df.columns and df["DATE"].dtype == pl.Utf8:
            df = df.with_columns(
                [pl.col("DATE").str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False)]
            )
        
        return df
    
    except Exception as e:
        logger.error("Error reading %s: %s", file_path, e)
        return None


def combine_station_data(
    station_id: str,
    merged_dir: Path,
    merged1_dir: Path,
    output_dir: Path,
    use_parquet: bool = False
) -> bool:
    """
    Combine data for a single station from both directories.
    
    Args:
        station_id: Station identifier
        merged_dir: Path to first merged directory
        merged1_dir: Path to second merged directory
        output_dir: Path to output directory
        use_parquet: If True, write as Parquet; if False, write as CSV.gz
        
    Returns:
        True if successful, False otherwise
    """
    files_to_merge = []
    
    # Check merged/
    csv_file = merged_dir / f"{station_id}.csv.gz"
    parquet_file = merged_dir / f"{station_id}.parquet"
    if parquet_file.exists():
        files_to_merge.append(parquet_file)
    elif csv_file.exists():
        files_to_merge.append(csv_file)
    
    # Check merged-1/
    csv_file1 = merged1_dir / f"{station_id}.csv.gz"
    parquet_file1 = merged1_dir / f"{station_id}.parquet"
    if parquet_file1.exists():
        files_to_merge.append(parquet_file1)
    elif csv_file1.exists():
        files_to_merge.append(csv_file1)
    
    if not files_to_merge:
        return False
    
    # Read all files for this station
    dfs = []
    for file_path in files_to_merge:
        df = read_station_file(file_path)
        if df is not None and df.shape[0] > 0:
            dfs.append(df)
    
    if not dfs:
        return False
    
    try:
        # Merge all dataframes
        merged_df = pl.concat(dfs, how="diagonal")
        
        # Deduplicate based on DATE and STATION (if STATION column exists)
        if "STATION" in merged_df.columns and "DATE" in merged_df.columns:
            merged_df = merged_df.unique(subset=["DATE", "STATION"], keep="first")
        elif "DATE" in merged_df.columns:
            merged_df = merged_df.unique(subset=["DATE"], keep="first")
        else:
            merged_df = merged_df.unique()
        
        # Sort by date if DATE column exists
        if "DATE" in merged_df.columns:
            merged_df = merged_df.sort("DATE")
        
        # Format DATE column back to string if needed
        if "DATE" in merged_df.columns and merged_df["DATE"].dtype == pl.Datetime:
            merged_df = merged_df.with_columns(
                [pl.col("DATE").dt.strftime("%Y-%m-%dT%H:%M:%S")]
            )
        
        # Cast all columns to strings for consistent output
        merged_df = merged_df.select(
            [pl.col(col).cast(pl.Utf8) for col in merged_df.columns]
        )
        
        # Write to output directory
        if use_parquet:
            output_file = output_dir / f"{station_id}.parquet"
            merged_df.write_parquet(output_file, compression="zstd")
        else:
            output_file = output_dir / f"{station_id}.csv.gz"
            with gzip.open(output_file, "wb") as f:
                merged_df.write_csv(f, quote_style="necessary")
        
        return True
    
    except Exception as e:
        logger.error("Error combining data for %s: %s", station_id, e)
        return False


def merge_metadata_databases(
    merged_dir: Path,
    merged1_dir: Path,
    output_dir: Path
) -> bool:
    """
    Merge metadata from both SQLite databases into a new combined database.
    
    Args:
        merged_dir: Path to first merged directory
        merged1_dir: Path to second merged directory
        output_dir: Path to output directory
        
    Returns:
        True if successful, False otherwise
    """
    db1_path = merged_dir / "wx_metadata.db"
    db2_path = merged1_dir / "wx_metadata.db"
    output_db_path = output_dir / "wx_metadata.db"
    
    # Check if databases exist
    if not db1_path.exists() and not db2_path.exists():
        logger.warning("No metadata databases found in either directory")
        return False
    
    try:
        logger.info("Merging metadata databases...")
        
        # Create output database
        output_db_path.parent.mkdir(parents=True, exist_ok=True)
        if output_db_path.exists():
            output_db_path.unlink()
        
        conn_out = sqlite3.connect(output_db_path)
        conn_out.execute("PRAGMA foreign_keys = ON")
        conn_out.execute("PRAGMA journal_mode = WAL")
        
        # Create schema
        conn_out.execute("""
            CREATE TABLE IF NOT EXISTS metadata_settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        conn_out.execute("""
            CREATE TABLE IF NOT EXISTS stations (
                station_id TEXT PRIMARY KEY,
                latitude TEXT,
                longitude TEXT,
                elevation TEXT,
                name TEXT,
                call_sign TEXT,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                -- High priority metadata
                first_observation_date TEXT,
                last_observation_date TEXT,
                total_observation_count INTEGER,
                tmp_completeness_pct REAL,
                dew_completeness_pct REAL,
                slp_completeness_pct REAL,
                wnd_completeness_pct REAL,
                vis_completeness_pct REAL,
                cig_completeness_pct REAL,
                tmp_mean REAL,
                tmp_std REAL,
                tmp_min REAL,
                tmp_max REAL,
                dew_mean REAL,
                dew_std REAL,
                slp_mean REAL,
                slp_std REAL,
                slp_min REAL,
                slp_max REAL,
                -- Medium priority metadata
                gap_count INTEGER,
                max_gap_duration_hours REAL,
                timezone TEXT,
                observation_frequency REAL
            )
        """)
        
        conn_out.execute("""
            CREATE TABLE IF NOT EXISTS year_counts (
                station_id TEXT,
                year TEXT,
                observation_count INTEGER,
                PRIMARY KEY (station_id, year),
                FOREIGN KEY (station_id) REFERENCES stations(station_id) ON DELETE CASCADE
            )
        """)
        
        conn_out.execute("CREATE INDEX IF NOT EXISTS idx_year_counts_year ON year_counts(year)")
        
        # Merge from first database
        if db1_path.exists():
            logger.info("Importing metadata from %s", db1_path)
            conn1 = sqlite3.connect(db1_path)

            # Get column names from db1 to handle schema differences
            cursor1 = conn1.execute("PRAGMA table_info(stations)")
            db1_columns = [row[1] for row in cursor1]

            # Get column mapping for output database
            cursor_out = conn_out.execute("PRAGMA table_info(stations)")
            out_columns = {row[1]: row[0] for row in cursor_out}

            # Build INSERT columns list (only include columns that exist in output schema)
            insert_cols = [col for col in db1_columns if col in out_columns]

            if insert_cols:
                # Read all rows from db1 and insert into output
                select_cols = ", ".join(insert_cols)
                cursor1 = conn1.execute(f"SELECT {select_cols} FROM stations")

                for row in cursor1:
                    placeholders = ", ".join(["?"] * len(insert_cols))
                    conn_out.execute(
                        f"INSERT OR IGNORE INTO stations ({', '.join(insert_cols)}) VALUES ({placeholders})",
                        row
                    )

            # Copy year counts (check if table exists first)
            try:
                cursor1 = conn1.execute("SELECT station_id, year, observation_count FROM year_counts")
                for row in cursor1:
                    conn_out.execute(
                        "INSERT OR REPLACE INTO year_counts (station_id, year, observation_count) VALUES (?, ?, ?)",
                        row
                    )
            except sqlite3.OperationalError as e:
                logger.warning("Could not copy year_counts from db1: %s", e)

            conn1.close()
        
        # Merge from second database
        if db2_path.exists():
            logger.info("Importing metadata from %s", db2_path)
            conn2 = sqlite3.connect(db2_path)
            
            # Get column names from db2 to handle schema differences
            cursor2 = conn2.execute("PRAGMA table_info(stations)")
            db2_columns = [row[1] for row in cursor2]
            
            # Build SELECT query with all available columns
            select_cols = ", ".join(db2_columns)
            cursor2 = conn2.execute(f"SELECT {select_cols} FROM stations")
            
            # Get column mapping for output database
            cursor_out = conn_out.execute("PRAGMA table_info(stations)")
            out_columns = {row[1]: row[0] for row in cursor_out}
            
            for row in cursor2:
                # Build dictionary of column values
                row_dict = dict(zip(db2_columns, row))
                station_id = row_dict.get("station_id")
                if not station_id:
                    continue
                
                # Build INSERT/UPDATE query dynamically
                # For basic metadata, prefer non-empty values from db2
                # For statistics, prefer more recent data (db2) or merge appropriately
                update_parts = []
                
                # Basic metadata fields
                for col in ["latitude", "longitude", "elevation", "name", "call_sign"]:
                    if col in row_dict and col in out_columns:
                        update_parts.append(f"{col} = CASE WHEN excluded.{col} != '' THEN excluded.{col} ELSE {col} END")
                
                # Date range - use earliest first, latest last
                if "first_observation_date" in row_dict and "first_observation_date" in out_columns:
                    update_parts.append("""
                        first_observation_date = CASE 
                            WHEN excluded.first_observation_date IS NOT NULL AND excluded.first_observation_date != '' 
                                 AND (first_observation_date IS NULL OR first_observation_date = '' 
                                      OR excluded.first_observation_date < first_observation_date)
                            THEN excluded.first_observation_date
                            ELSE first_observation_date
                        END
                    """)
                
                if "last_observation_date" in row_dict and "last_observation_date" in out_columns:
                    update_parts.append("""
                        last_observation_date = CASE 
                            WHEN excluded.last_observation_date IS NOT NULL AND excluded.last_observation_date != '' 
                                 AND (last_observation_date IS NULL OR last_observation_date = '' 
                                      OR excluded.last_observation_date > last_observation_date)
                            THEN excluded.last_observation_date
                            ELSE last_observation_date
                        END
                    """)
                
                # Total count - sum if both exist
                if "total_observation_count" in row_dict and "total_observation_count" in out_columns:
                    update_parts.append("""
                        total_observation_count = COALESCE(total_observation_count, 0) + COALESCE(excluded.total_observation_count, 0)
                    """)
                
                # Statistics - prefer non-null values, or average if both exist
                stat_fields = [
                    "tmp_completeness_pct", "dew_completeness_pct", "slp_completeness_pct",
                    "wnd_completeness_pct", "vis_completeness_pct", "cig_completeness_pct",
                    "tmp_mean", "tmp_std", "tmp_min", "tmp_max",
                    "dew_mean", "dew_std",
                    "slp_mean", "slp_std", "slp_min", "slp_max",
                    "gap_count", "max_gap_duration_hours", "timezone", "observation_frequency"
                ]
                
                for col in stat_fields:
                    if col in row_dict and col in out_columns:
                        # Prefer non-null value, or use existing if both are null
                        update_parts.append(f"""
                            {col} = CASE 
                                WHEN excluded.{col} IS NOT NULL THEN excluded.{col}
                                ELSE {col}
                            END
                        """)
                
                # Build INSERT columns and values (only include columns that exist in output schema)
                insert_cols = [col for col in db2_columns if col in out_columns]
                insert_vals = [row_dict.get(col) for col in insert_cols]
                
                if update_parts and len(insert_cols) > 0:
                    query = f"""
                        INSERT INTO stations ({', '.join(insert_cols)})
                        VALUES ({', '.join(['?'] * len(insert_cols))})
                        ON CONFLICT(station_id) DO UPDATE SET
                            {', '.join(update_parts)}
                    """
                    conn_out.execute(query, insert_vals)
                elif len(insert_cols) > 0:
                    # Fallback to simple insert/ignore
                    query = f"""
                        INSERT OR IGNORE INTO stations ({', '.join(insert_cols)})
                        VALUES ({', '.join(['?'] * len(insert_cols))})
                    """
                    conn_out.execute(query, insert_vals)
            
            # Copy year counts (will replace if year exists, add if new)
            try:
                cursor2 = conn2.execute("SELECT station_id, year, observation_count FROM year_counts")
                for row in cursor2:
                    conn_out.execute(
                        "INSERT OR REPLACE INTO year_counts (station_id, year, observation_count) VALUES (?, ?, ?)",
                        row
                    )
            except sqlite3.OperationalError as e:
                logger.warning("Could not copy year_counts from db2: %s", e)

            conn2.close()
        
        # Set schema version
        conn_out.execute(
            "INSERT OR REPLACE INTO metadata_settings (key, value) VALUES (?, ?)",
            ("schema_version", "2")
        )
        
        conn_out.commit()
        conn_out.close()
        
        logger.info("Metadata database merged successfully: %s", output_db_path)
        return True
        
    except Exception as e:
        logger.error("Error merging metadata databases: %s", e, exc_info=True)
        return False


def export_metadata_to_csv(output_dir: Path) -> bool:
    """
    Export merged metadata to CSV format for compatibility.
    
    Args:
        output_dir: Path to output directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from metadata_manager import export_to_csv, DB_PATH, LEGACY_CSV_PATH
        
        # Temporarily set the paths to point to output directory
        original_db_path = DB_PATH
        original_csv_path = LEGACY_CSV_PATH
        
        # Use output directory for export
        output_db = output_dir / "wx_metadata.db"
        output_csv = output_dir / "wx_info.csv"
        
        if not output_db.exists():
            logger.warning("Metadata database not found at %s", output_db)
            return False
        
        # Import the metadata manager and temporarily modify paths
        import metadata_manager
        metadata_manager.DB_PATH = output_db
        metadata_manager.LEGACY_CSV_PATH = output_csv
        
        # Export
        result = export_to_csv(output_csv)
        
        # Restore original paths
        metadata_manager.DB_PATH = original_db_path
        metadata_manager.LEGACY_CSV_PATH = original_csv_path
        
        if result:
            logger.info("Exported metadata to %s", output_csv)
        
        return result
        
    except Exception as e:
        logger.error("Error exporting metadata to CSV: %s", e, exc_info=True)
        return False


def combine_datasets(
    merged_dir: Path,
    merged1_dir: Path,
    output_dir: Path,
    use_parquet: bool = False
) -> None:
    """
    Combine two merged datasets into one, including metadata.
    
    Args:
        merged_dir: Path to first merged directory (merged/)
        merged1_dir: Path to second merged directory (merged-1/)
        output_dir: Path to output directory (merged-combined/)
        use_parquet: If True, write as Parquet; if False, write as CSV.gz
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all station IDs
    all_stations = get_all_station_ids(merged_dir, merged1_dir)
    
    if not all_stations:
        logger.warning("No stations found in either directory")
        return
    
    # Process each station
    success_count = 0
    failed_count = 0
    
    format_str = "Parquet" if use_parquet else "CSV.gz"
    logger.info("Combining datasets, output format: %s", format_str)
    
    for station_id in tqdm(all_stations, desc="Combining stations", unit="station"):
        if combine_station_data(station_id, merged_dir, merged1_dir, output_dir, use_parquet):
            success_count += 1
        else:
            failed_count += 1
    
    logger.info("Combined %d stations successfully, %d failed", success_count, failed_count)
    
    # Merge metadata databases
    logger.info("Merging metadata databases...")
    if merge_metadata_databases(merged_dir, merged1_dir, output_dir):
        logger.info("Metadata databases merged successfully")
        
        # Compute statistics for all stations in output directory
        logger.info("Computing statistics for all stations...")
        compute_all_station_statistics(output_dir)
        
        # Export to CSV for compatibility
        logger.info("Exporting metadata to CSV...")
        export_metadata_to_csv(output_dir)
    else:
        logger.warning("Metadata merge failed or skipped")
    
    logger.info("Output directory: %s", output_dir)


def compute_all_station_statistics(output_dir: Path):
    """
    Compute statistics for all station files in the output directory.

    Args:
        output_dir: Path to output directory containing station files
    """
    import metadata_manager
    from metadata_manager import update_station_statistics

    # Temporarily set the DB path to point to output directory
    original_db_path = metadata_manager.DB_PATH
    original_csv_path = metadata_manager.LEGACY_CSV_PATH
    original_is_initialized = metadata_manager._is_initialized

    metadata_manager.DB_PATH = output_dir / "wx_metadata.db"
    metadata_manager.LEGACY_CSV_PATH = output_dir / "wx_info.csv"
    metadata_manager._is_initialized = False  # Force re-initialization with new path

    try:
        # Get all station files
        station_files = list(output_dir.glob("*.csv.gz")) + list(output_dir.glob("*.parquet"))

        if not station_files:
            logger.warning("No station files found in %s", output_dir)
            return

        logger.info("Computing statistics for %d stations...", len(station_files))

        success_count = 0
        failed_count = 0

        for station_file in tqdm(station_files, desc="Computing statistics", unit="station"):
            try:
                station_id = station_file.stem.replace(".csv", "")
                update_station_statistics(station_id, station_file)
                success_count += 1
            except Exception as e:
                logger.error("Error computing statistics for %s: %s", station_file.name, e)
                failed_count += 1

        logger.info("Computed statistics for %d stations, %d failed", success_count, failed_count)
    finally:
        # Restore original paths
        metadata_manager.DB_PATH = original_db_path
        metadata_manager.LEGACY_CSV_PATH = original_csv_path
        metadata_manager._is_initialized = original_is_initialized


def main():
    """Main entry point for the combine_datasets script."""
    parser = argparse.ArgumentParser(
        description="Combine two merged datasets into one"
    )
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=DATA_DIR / "merged",
        help="Path to first merged directory (default: data/merged)",
    )
    parser.add_argument(
        "--merged1-dir",
        type=Path,
        default=DATA_DIR / "merged-1",
        help="Path to second merged directory (default: data/merged-1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "merged-combined",
        help="Path to output directory (default: data/merged-combined)",
    )
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="Write output as Parquet files instead of CSV.gz",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only run metadata merge and statistics calculation (skip combining station data)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only compute statistics for existing station files (requires --output-dir to exist)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Handle stats-only mode
    if args.stats_only:
        if not args.output_dir.exists():
            logger.error("Output directory does not exist: %s", args.output_dir)
            return

        logger.info("Running statistics calculation only on %s", args.output_dir)
        compute_all_station_statistics(args.output_dir)
        export_metadata_to_csv(args.output_dir)
        logger.info("Statistics calculation complete")
        return

    # Handle metadata-only mode
    if args.metadata_only:
        if not args.output_dir.exists():
            logger.error("Output directory does not exist: %s", args.output_dir)
            return

        logger.info("Running metadata-only mode on %s", args.output_dir)

        # Merge metadata databases
        if merge_metadata_databases(args.merged_dir, args.merged1_dir, args.output_dir):
            logger.info("Metadata databases merged successfully")

            # Compute statistics for all stations in output directory
            logger.info("Computing statistics for all stations...")
            compute_all_station_statistics(args.output_dir)

            # Export to CSV for compatibility
            logger.info("Exporting metadata to CSV...")
            export_metadata_to_csv(args.output_dir)
        else:
            logger.warning("Metadata merge failed")

        logger.info("Metadata-only mode complete")
        return

    # Validate input directories for full combine
    if not args.merged_dir.exists():
        logger.error("Merged directory does not exist: %s", args.merged_dir)
        return

    if not args.merged1_dir.exists():
        logger.error("Merged-1 directory does not exist: %s", args.merged1_dir)
        return
    
    # Combine datasets
    combine_datasets(
        args.merged_dir,
        args.merged1_dir,
        args.output_dir,
        use_parquet=args.parquet
    )


if __name__ == "__main__":
    main()
