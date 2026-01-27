#!/usr/bin/env python3
"""CLI for preprocessing weather data for imputation model training.

This script loads selected stations, filters and normalizes their data,
applies train/val/test splits, and saves processed datasets.

Usage:
    python preprocess.py                                    # Use default settings
    python preprocess.py --stations-file path/to/stations.json
    python preprocess.py --output-dir data/processed
    python preprocess.py --split-strategy spatial          # spatial, temporal, hybrid, or simulated
"""

import json
import logging
from pathlib import Path

import polars as pl
import typer
from rich.console import Console
from rich.progress import track

from weather_imputation.config.data import (
    SplitConfig,
)
from weather_imputation.config.paths import PROCESSED_DIR
from weather_imputation.data.ghcnh_loader import (
    TIER1_VARIABLES,
    extract_tier1_variables,
    filter_by_quality_flags,
    load_station_all_years,
)
from weather_imputation.data.normalization import Normalizer
from weather_imputation.data.splits import create_split
from weather_imputation.utils.progress import (
    print_error,
    print_info,
    print_success,
    print_warning,
)

app = typer.Typer(help="Preprocess weather data for imputation model training.")
console = Console()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def preprocess(
    stations_file: Path = typer.Option(
        PROCESSED_DIR / "selected_stations.json",
        "--stations-file",
        "-s",
        help="Path to JSON file with selected station IDs",
    ),
    metadata_file: Path = typer.Option(
        PROCESSED_DIR / "metadata_cleaned.parquet",
        "--metadata-file",
        "-m",
        help="Path to metadata parquet file",
    ),
    output_dir: Path = typer.Option(
        PROCESSED_DIR,
        "--output-dir",
        "-o",
        help="Output directory for processed datasets",
    ),
    split_strategy: str = typer.Option(
        "simulated",
        "--split-strategy",
        help="Split strategy: spatial, temporal, hybrid, or simulated",
    ),
    normalization_method: str = typer.Option(
        "zscore",
        "--normalization",
        "-n",
        help="Normalization method: zscore, minmax, or none",
    ),
    variables: str = typer.Option(
        ",".join(TIER1_VARIABLES),
        "--variables",
        "-v",
        help="Comma-separated list of variables to include",
    ),
    exclude_suspects: bool = typer.Option(
        False,
        "--exclude-suspects",
        help="Exclude suspect quality flags (more aggressive filtering)",
    ),
    train_ratio: float = typer.Option(
        0.7,
        "--train-ratio",
        help="Training set ratio (for spatial/temporal splits)",
        min=0.1,
        max=0.9,
    ),
    val_ratio: float = typer.Option(
        0.15,
        "--val-ratio",
        help="Validation set ratio (for spatial/temporal splits)",
        min=0.05,
        max=0.4,
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing processed files",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet output"),
) -> None:
    """Preprocess weather data for imputation model training.

    This command:
    1. Loads selected stations from JSON file
    2. Loads and filters station data (quality flags, variables)
    3. Normalizes data per-station
    4. Applies train/val/test splits
    5. Saves processed datasets as parquet files
    """
    setup_logging(verbose, quiet)

    # Validate inputs
    if not stations_file.exists():
        print_error(f"Stations file not found: {stations_file}")
        print_info(
            "Generate this file using: marimo edit notebooks/01_station_exploration.py"
        )
        raise typer.Exit(1)

    if not metadata_file.exists():
        print_error(f"Metadata file not found: {metadata_file}")
        print_info("Generate metadata using: uv run python src/scripts/compute_metadata.py")
        raise typer.Exit(1)

    if split_strategy not in ["spatial", "temporal", "hybrid", "simulated"]:
        print_error(f"Invalid split strategy: {split_strategy}")
        print_info("Valid options: spatial, temporal, hybrid, simulated")
        raise typer.Exit(1)

    if normalization_method not in ["zscore", "minmax", "none"]:
        print_error(f"Invalid normalization method: {normalization_method}")
        print_info("Valid options: zscore, minmax, none")
        raise typer.Exit(1)

    if train_ratio + val_ratio >= 1.0:
        print_error("train_ratio + val_ratio must be < 1.0")
        raise typer.Exit(1)

    # Parse variables
    variable_list = [v.strip() for v in variables.split(",")]

    # Check if output files already exist
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    if not force and all(p.exists() for p in [train_path, val_path, test_path]):
        print_warning("Processed files already exist. Use --force to overwrite.")
        raise typer.Exit(0)

    print_info("Starting preprocessing pipeline...")
    print_info(f"Split strategy: {split_strategy}")
    print_info(f"Normalization: {normalization_method}")
    print_info(f"Variables: {', '.join(variable_list)}")

    # Load selected stations
    print_info(f"Loading stations from: {stations_file}")
    with open(stations_file) as f:
        station_data = json.load(f)

    # Handle different JSON formats (list vs dict)
    if isinstance(station_data, list):
        station_ids = station_data
    elif isinstance(station_data, dict) and "stations" in station_data:
        station_ids = station_data["stations"]
    else:
        print_error("Invalid stations file format. Expected list or dict with 'stations' key.")
        raise typer.Exit(1)

    print_info(f"Loaded {len(station_ids)} stations")

    # Load metadata
    print_info(f"Loading metadata from: {metadata_file}")
    metadata = pl.read_parquet(metadata_file)

    # Filter metadata to selected stations
    metadata = metadata.filter(pl.col("station_id").is_in(station_ids))
    print_info(f"Filtered metadata to {len(metadata)} stations")

    # Create split configuration
    split_config = SplitConfig(
        strategy=split_strategy,  # type: ignore[arg-type]
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    # Apply split to get train/val/test station IDs
    print_info(f"Applying {split_strategy} split...")
    train_ids, val_ids, test_ids = create_split(metadata, split_config)

    print_info("Split results:")
    print_info(f"  Train: {len(train_ids)} stations")
    print_info(f"  Val: {len(val_ids)} stations")
    print_info(f"  Test: {len(test_ids)} stations")

    # Process each split
    for split_name, split_ids, output_path in [
        ("train", train_ids, train_path),
        ("val", val_ids, val_path),
        ("test", test_ids, test_path),
    ]:
        print_info(f"\nProcessing {split_name} split ({len(split_ids)} stations)...")

        all_station_data = []
        skipped_stations = []

        for station_id in track(
            split_ids, description=f"Loading {split_name} stations..."
        ):
            try:
                # Get station metadata
                station_meta = metadata.filter(
                    pl.col("station_id") == station_id
                ).to_dicts()[0]

                # Determine year range from metadata
                first_year = station_meta["first_observation"].year
                last_year = station_meta["last_observation"].year

                # Load station data
                df = load_station_all_years(
                    station_id=station_id,
                    start_year=first_year,
                    end_year=last_year,
                )

                if df is None or len(df) == 0:
                    logger.warning(f"No data loaded for station {station_id}")
                    skipped_stations.append(station_id)
                    continue

                # Extract Tier 1 variables
                df = extract_tier1_variables(df, variables=variable_list)

                # Apply quality filtering
                df = filter_by_quality_flags(
                    df,
                    exclude_suspects=exclude_suspects,
                    variables=variable_list,
                )

                # Add station metadata columns
                df = df.with_columns([
                    pl.lit(station_id).alias("station_id"),
                    pl.lit(station_meta["latitude"]).alias("latitude"),
                    pl.lit(station_meta["longitude"]).alias("longitude"),
                    pl.lit(station_meta.get("elevation")).alias("elevation"),
                ])

                # Normalize data if requested
                if normalization_method != "none":
                    normalizer = Normalizer(method=normalization_method)  # type: ignore[arg-type]

                    # Fit normalizer on observed values only
                    value_cols = [v for v in variable_list if v in df.columns]
                    if value_cols:
                        normalizer.fit(df.select(value_cols))

                        # Transform variables
                        normalized_df = normalizer.transform(df.select(value_cols))

                        # Replace normalized columns in original dataframe
                        for col in value_cols:
                            df = df.with_columns(normalized_df.select(col))

                        # Save normalization stats for later inverse transform
                        df = df.with_columns([
                            pl.lit(json.dumps(normalizer.stats_)).alias("normalization_stats")
                        ])

                all_station_data.append(df)

            except Exception as e:
                logger.error(f"Error processing station {station_id}: {e}")
                skipped_stations.append(station_id)
                continue

        if skipped_stations:
            print_warning(
                f"Skipped {len(skipped_stations)} stations due to errors"
            )

        if not all_station_data:
            print_error(f"No data to save for {split_name} split")
            continue

        # Concatenate all station data
        print_info(f"Concatenating {len(all_station_data)} station datasets...")
        combined_df = pl.concat(all_station_data, how="vertical_relaxed")

        # Save to parquet
        print_info(f"Saving {split_name} split to: {output_path}")
        combined_df.write_parquet(output_path)

        # Print statistics
        print_success(
            f"{split_name.capitalize()} split saved: {len(combined_df)} rows, "
            f"{combined_df['station_id'].n_unique()} stations"
        )

    print_success("\nPreprocessing complete!")
    print_info(f"Processed files saved to: {output_dir}")
    print_info("Next steps:")
    print_info("  1. Train models: uv run python src/scripts/train.py")
    print_info("  2. Evaluate models: uv run python src/scripts/evaluate.py")


if __name__ == "__main__":
    app()
