#!/usr/bin/env python3
"""CLI for cleaning and deduplicating station metadata.

Usage:
    python clean_metadata.py clean                      # Run cleaning pipeline
    python clean_metadata.py clean --dry-run            # Preview changes
    python clean_metadata.py duplicates                 # List duplicate stations
    python clean_metadata.py report                     # Show cleaning report
"""

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from weather_imputation.config.paths import (
    CLEANING_LOG_TXT,
    CLEANING_REPORT_JSON,
    METADATA_CLEANED_CSV,
    METADATA_CLEANED_PARQUET,
    METADATA_PARQUET,
)
from weather_imputation.data.cleaning import (
    CleaningLog,
    identify_duplicate_stations,
    run_cleaning_pipeline,
    validate_coordinates,
)
from weather_imputation.data.metadata import load_metadata, save_metadata
from weather_imputation.utils.progress import (
    print_error,
    print_info,
    print_success,
    print_summary_table,
    print_warning,
    status_spinner,
)

app = typer.Typer(help="Clean and deduplicate station metadata.")
console = Console()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


@app.command()
def clean(
    input_file: Path | None = typer.Option(
        None,
        "--input", "-i",
        help="Input metadata parquet file",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output", "-o",
        help="Output cleaned metadata parquet file",
    ),
    export_csv: bool = typer.Option(
        True,
        "--csv/--no-csv",
        help="Export cleaned metadata to CSV",
    ),
    csv_output: Path | None = typer.Option(
        None,
        "--csv-path",
        help="Custom CSV output path (default: metadata_cleaned.csv)",
    ),
    log_output: Path | None = typer.Option(
        None,
        "--log",
        help="Path for detailed cleaning log (default: clean_log.txt)",
    ),
    merge_duplicates: bool = typer.Option(
        True,
        "--merge-duplicates/--no-merge-duplicates",
        help="Merge duplicate stations",
    ),
    merge_strategy: str = typer.Option(
        "prefer_longer",
        "--strategy",
        help="Merge strategy: prefer_longer (more observations) or prefer_recent (latest data)",
    ),
    fill_coordinates: bool = typer.Option(
        True,
        "--fill-coords/--no-fill-coords",
        help="Lookup missing lat/lon/elevation",
    ),
    validate_coords: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate coordinate ranges",
    ),
    clean_names: bool = typer.Option(
        True,
        "--clean-names/--no-clean-names",
        help="Clean and normalize station names",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run", "-n",
        help="Show what would change without saving",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Clean metadata: deduplicate stations, fill missing coordinates."""
    setup_logging(verbose)

    # Determine input/output paths
    input_path = input_file or METADATA_PARQUET
    output_path = output_file or METADATA_CLEANED_PARQUET
    csv_path = csv_output if csv_output else METADATA_CLEANED_CSV
    log_path = log_output or CLEANING_LOG_TXT

    # Load metadata
    if not input_path.exists():
        print_error(f"Input file not found: {input_path}")
        print_info("Run 'compute_metadata.py' first to generate metadata.")
        raise typer.Exit(1)

    print_info(f"Loading metadata from {input_path}")
    metadata = load_metadata(cleaned=False)

    if metadata is None or len(metadata) == 0:
        print_error("No metadata found in input file.")
        raise typer.Exit(1)

    print_info(f"Loaded {len(metadata)} stations")

    # Create cleaning log
    cleaning_log = CleaningLog()

    # Run cleaning pipeline
    console.print("\n[bold]Running cleaning pipeline...[/bold]")

    with status_spinner("Cleaning metadata..."):
        cleaned, report = run_cleaning_pipeline(
            metadata,
            merge_duplicates=merge_duplicates,
            merge_strategy=merge_strategy,
            fill_coordinates=fill_coordinates,
            validate_coords=validate_coords,
            clean_names=clean_names,
            log=cleaning_log,
        )

    # Show report
    console.print()
    print_summary_table("Cleaning Results", {
        "Original stations": report.original_station_count,
        "Cleaned stations": report.cleaned_station_count,
        "Duplicates found": report.duplicates_found,
        "Duplicates merged": report.duplicates_merged,
        "Coordinates filled": report.coordinates_filled,
        "Coordinates valid": report.coordinates_validated,
        "Coordinates invalid": report.coordinates_invalid,
        "Names cleaned": report.names_cleaned,
    })

    if dry_run:
        print_warning("\n[DRY RUN] No changes saved.")
        print_info(f"Would save to: {output_path}")
        if export_csv:
            print_info(f"Would export CSV to: {csv_path}")
        print_info(f"Would save log to: {log_path}")
    else:
        # Save cleaned metadata
        save_metadata(cleaned, cleaned=True, export_csv=export_csv, csv_path=csv_path)
        report.save()
        cleaning_log.save(log_path)
        print_success(f"\nSaved cleaned metadata to {output_path}")
        if export_csv:
            print_success(f"Exported CSV to {csv_path}")
        print_success(f"Saved cleaning log to {log_path}")
        print_success(f"Saved report to {CLEANING_REPORT_JSON}")


@app.command()
def duplicates(
    input_file: Path | None = typer.Option(
        None,
        "--input", "-i",
        help="Input metadata parquet file",
    ),
    threshold: float = typer.Option(
        0.01,
        "--threshold", "-t",
        help="Location threshold in degrees for duplicate detection",
    ),
    limit: int = typer.Option(
        50,
        "--limit", "-n",
        help="Maximum number of duplicates to show",
    ),
) -> None:
    """List potential duplicate stations for review."""
    input_path = input_file or METADATA_PARQUET

    if not input_path.exists():
        print_error(f"Input file not found: {input_path}")
        raise typer.Exit(1)

    metadata = load_metadata(cleaned=False)
    if metadata is None:
        print_error("Could not load metadata.")
        raise typer.Exit(1)

    print_info(f"Checking {len(metadata)} stations for duplicates...")

    with status_spinner("Identifying duplicates..."):
        duplicates_df = identify_duplicate_stations(metadata, location_threshold=threshold)

    if len(duplicates_df) == 0:
        print_success("No duplicate stations found.")
        return

    # Create table
    table = Table(title=f"Potential Duplicate Stations ({len(duplicates_df)} found)")
    table.add_column("Station", style="cyan")
    table.add_column("Duplicate Of", style="yellow")
    table.add_column("Confidence", justify="right")
    table.add_column("Reason")

    for row in duplicates_df.head(limit).iter_rows(named=True):
        table.add_row(
            row["station_id"],
            row["duplicate_of"],
            f"{row['confidence_score']:.2f}",
            row["reason"],
        )

    console.print(table)

    if len(duplicates_df) > limit:
        console.print(f"\n[dim]Showing {limit} of {len(duplicates_df)} duplicates[/dim]")

    console.print("\n[dim]Use 'clean --merge-duplicates' to merge these stations[/dim]")


@app.command()
def report(
    report_file: Path | None = typer.Option(
        None,
        "--file", "-f",
        help="Path to cleaning report JSON",
    ),
) -> None:
    """Show cleaning report and statistics."""
    report_path = report_file or CLEANING_REPORT_JSON

    if not report_path.exists():
        print_error(f"Report file not found: {report_path}")
        print_info("Run 'clean' first to generate a report.")
        raise typer.Exit(1)

    import json
    with open(report_path) as f:
        report_data = json.load(f)

    print_summary_table("Cleaning Report", report_data)


@app.command()
def validate(
    input_file: Path | None = typer.Option(
        None,
        "--input", "-i",
        help="Input metadata parquet file",
    ),
) -> None:
    """Validate coordinates in metadata and show invalid entries."""
    input_path = input_file or METADATA_PARQUET

    if not input_path.exists():
        print_error(f"Input file not found: {input_path}")
        raise typer.Exit(1)

    metadata = load_metadata(cleaned=False)
    if metadata is None:
        print_error("Could not load metadata.")
        raise typer.Exit(1)

    print_info(f"Validating {len(metadata)} stations...")

    with status_spinner("Validating coordinates..."):
        validated, valid_count, invalid_count = validate_coordinates(metadata)

    print_summary_table("Validation Results", {
        "Total stations": len(metadata),
        "Valid coordinates": valid_count,
        "Invalid coordinates": invalid_count,
    })

    if invalid_count > 0:
        # Show invalid entries
        import polars as pl
        invalid = validated.filter(~pl.col("coord_valid"))

        table = Table(title="Invalid Coordinate Entries")
        table.add_column("Station ID", style="cyan")
        table.add_column("Latitude", justify="right")
        table.add_column("Longitude", justify="right")
        table.add_column("Elevation", justify="right")

        for row in invalid.head(20).iter_rows(named=True):
            table.add_row(
                row.get("station_id", ""),
                str(row.get("latitude", "N/A")),
                str(row.get("longitude", "N/A")),
                str(row.get("elevation", "N/A")),
            )

        console.print(table)

        if invalid_count > 20:
            console.print(f"\n[dim]Showing 20 of {invalid_count} invalid entries[/dim]")


if __name__ == "__main__":
    app()
