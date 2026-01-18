#!/usr/bin/env python3
"""
Repair script to fix stations with data loss from combine_datasets.py datetime bug.
This script identifies stations with < 10 rows and re-combines them using the fixed logic.
"""
import argparse
import gzip
import logging
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import polars as pl

from config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def read_station_file(file_path: Path) -> Optional[pl.DataFrame]:
    """Read a station file with fixed datetime parsing."""
    try:
        if file_path.suffix == ".gz" or file_path.suffixes == [".csv", ".gz"]:
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
            df = pl.read_parquet(file_path)
        else:
            return None

        # Fixed datetime parsing - handle both with and without microseconds
        if "DATE" in df.columns and df["DATE"].dtype == pl.Utf8:
            df = df.with_columns(
                [pl.col("DATE").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False)
                 .fill_null(pl.col("DATE").str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False))]
            )

        return df

    except Exception as e:
        logger.error("Error reading %s: %s", file_path, e)
        return None


def repair_station(
    station_id: str,
    merged_dir: Path,
    merged1_dir: Path,
    output_dir: Path,
    use_parquet: bool = True
) -> bool:
    """Re-combine a single station with fixed datetime parsing."""
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

    # Read all files
    dfs = []
    for file_path in files_to_merge:
        df = read_station_file(file_path)
        if df is not None and df.shape[0] > 0:
            dfs.append(df)

    if not dfs:
        return False

    try:
        # Merge
        merged_df = pl.concat(dfs, how="diagonal")

        # Deduplicate
        if "STATION" in merged_df.columns and "DATE" in merged_df.columns:
            merged_df = merged_df.unique(subset=["DATE", "STATION"], keep="first")
        elif "DATE" in merged_df.columns:
            merged_df = merged_df.unique(subset=["DATE"], keep="first")
        else:
            merged_df = merged_df.unique()

        # Sort by date
        if "DATE" in merged_df.columns:
            merged_df = merged_df.sort("DATE")

        # Format DATE back to string
        if "DATE" in merged_df.columns and merged_df["DATE"].dtype == pl.Datetime:
            merged_df = merged_df.with_columns(
                [pl.col("DATE").dt.strftime("%Y-%m-%dT%H:%M:%S")]
            )

        # Cast all to strings
        merged_df = merged_df.select(
            [pl.col(col).cast(pl.Utf8) for col in merged_df.columns]
        )

        # Write output
        if use_parquet:
            output_file = output_dir / f"{station_id}.parquet"
            merged_df.write_parquet(output_file, compression="zstd")
        else:
            output_file = output_dir / f"{station_id}.csv.gz"
            with gzip.open(output_file, "wb") as f:
                merged_df.write_csv(f, quote_style="necessary")

        return True

    except Exception as e:
        logger.error("Error repairing %s: %s", station_id, e)
        return False


def find_broken_stations(output_dir: Path, threshold: int = 10) -> list:
    """Find stations with suspiciously low row counts."""
    broken = []

    logger.info("Scanning for broken stations (this may take a while)...")
    parquet_files = list(output_dir.glob("*.parquet"))

    for parquet_file in tqdm(parquet_files, desc="Scanning"):
        station_id = parquet_file.stem
        try:
            # Use scan_parquet for efficiency - just get row count
            df = pl.scan_parquet(parquet_file).select(pl.len()).collect()
            row_count = df.item()
            if row_count < threshold:
                broken.append(station_id)
        except Exception as e:
            logger.warning("Error scanning %s: %s", station_id, e)
            broken.append(station_id)

    return broken


def main():
    parser = argparse.ArgumentParser(description="Repair stations with data loss")
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=DATA_DIR / "merged",
        help="Path to first merged directory",
    )
    parser.add_argument(
        "--merged1-dir",
        type=Path,
        default=DATA_DIR / "merged-1",
        help="Path to second merged directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "merged-stations",
        help="Path to output directory to repair",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Stations with fewer rows than this will be repaired",
    )
    parser.add_argument(
        "--stations",
        type=str,
        help="Comma-separated list of station IDs to repair (skip scan)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan and report, don't repair",
    )

    args = parser.parse_args()

    # Get stations to repair
    if args.stations:
        broken_stations = [s.strip() for s in args.stations.split(",")]
        logger.info("Will repair %d specified stations", len(broken_stations))
    else:
        broken_stations = find_broken_stations(args.output_dir, args.threshold)
        logger.info("Found %d stations with < %d rows", len(broken_stations), args.threshold)

    if args.dry_run:
        logger.info("Dry run - not repairing")
        for s in broken_stations[:20]:
            print(f"  {s}")
        if len(broken_stations) > 20:
            print(f"  ... and {len(broken_stations) - 20} more")
        return

    # Repair stations
    success = 0
    failed = 0

    for station_id in tqdm(broken_stations, desc="Repairing"):
        if repair_station(station_id, args.merged_dir, args.merged1_dir, args.output_dir, use_parquet=True):
            success += 1
        else:
            failed += 1
            logger.warning("Failed to repair %s", station_id)

    logger.info("Repaired %d stations, %d failed", success, failed)


if __name__ == "__main__":
    main()
