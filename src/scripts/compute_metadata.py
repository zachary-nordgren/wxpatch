#!/usr/bin/env python3
"""CLI for computing station metadata from downloaded GHCNh parquet files.

Usage:
    python compute_metadata.py                          # Compute metadata for all stations/years
    python compute_metadata.py --years 2023,2024        # Specific years
    python compute_metadata.py --years 2020:2024        # Year range
    python compute_metadata.py --stations USW00003046   # Specific stations
    python compute_metadata.py --force                  # Recompute even if exists
"""

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from weather_imputation.config.paths import METADATA_PARQUET, PROCESSED_DIR
from weather_imputation.data.metadata import (
    compute_all_metadata,
    compute_all_metadata_incremental,
    load_metadata,
    save_metadata,
    enrich_metadata_from_station_list,
)
from weather_imputation.utils.parsing import parse_year_filter, parse_station_filter
from weather_imputation.utils.progress import (
    create_processing_progress,
    print_success,
    print_warning,
    print_info,
    print_summary_table,
)

app = typer.Typer(help="Compute station metadata from GHCNh parquet files.")
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
def compute(
    years: Optional[str] = typer.Option(
        None,
        "--years", "-y",
        help="Year filter (e.g., '2020,2021' or '2010:2020')",
    ),
    stations: Optional[str] = typer.Option(
        None,
        "--stations", "-s",
        help="Station filter (comma-separated station IDs)",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Recompute all stations even if metadata exists",
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental", "-i",
        help="Only recompute metadata for stations with modified files",
    ),
    export_csv: bool = typer.Option(
        True,
        "--csv/--no-csv",
        help="Export metadata to CSV",
    ),
    workers: int = typer.Option(
        10,
        "--workers", "-w",
        help="Number of parallel workers for station processing (default: 4, use 1 for sequential)",
        min=1,
        max=32,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet output"),
) -> None:
    """Compute station metadata from GHCNh parquet files."""
    setup_logging(verbose, quiet)

    # Parse filters
    year_filter = parse_year_filter(years) if years else None
    station_filter = parse_station_filter(stations) if stations else None

    if year_filter:
        print_info(f"Year filter: {year_filter}")
    if station_filter:
        print_info(f"Station filter: {station_filter[:5]}{'...' if len(station_filter) > 5 else ''}")

    # Handle existing metadata
    if METADATA_PARQUET.exists() and not force and not incremental:
        print_warning(f"Metadata file already exists: {METADATA_PARQUET}")
        print_info("Use --force to recompute all, or --incremental to update only changed stations")
        raise typer.Exit(0)

    # Create output directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if incremental:
        # Incremental mode: only compute for stations with modified files
        console.print("\n[bold]Computing station metadata (incremental mode)...[/bold]")

        with create_processing_progress() as progress:
            task_id = progress.add_task("Computing metadata", total=None)

            def progress_callback(current: int, total: int, station: str) -> None:
                progress.update(task_id, total=total, completed=current, description=f"Processing {station}")

            metadata_df, updated_count, unchanged_count = compute_all_metadata_incremental(
                year_filter=year_filter,
                station_filter=station_filter,
                progress_callback=progress_callback,
                max_workers=workers,
            )

        if len(metadata_df) == 0:
            print_warning("No metadata computed. Check that parquet files exist.")
            raise typer.Exit(1)

        if updated_count == 0:
            print_info("No stations needed updating - all metadata is current")
            raise typer.Exit(0)

        # Enrich with NOAA station inventory (lat/lon/elevation, WMO ID, ICAO, etc.)
        console.print("\n[bold]Enriching metadata from NOAA station inventory...[/bold]")
        metadata_df = enrich_metadata_from_station_list(metadata_df)

        # Save metadata
        save_metadata(metadata_df, cleaned=False, export_csv=export_csv)

        # Print summary
        no_hourly_count = (metadata_df["total_observation_count"] == 0).sum()
        print_success(f"Updated metadata for {updated_count} stations ({unchanged_count} unchanged)")
        print_summary_table("Metadata Summary", {
            "Stations updated": updated_count,
            "Stations unchanged": unchanged_count,
            "Total stations": len(metadata_df),
            "Stations with hourly data": len(metadata_df) - no_hourly_count,
            "Stations without hourly data": no_hourly_count,
            "Output file": str(METADATA_PARQUET),
            "CSV export": str(export_csv),
        })
    else:
        # Full recompute mode
        console.print("\n[bold]Computing station metadata...[/bold]")

        with create_processing_progress() as progress:
            task_id = progress.add_task("Computing metadata", total=None)

            def progress_callback(current: int, total: int, station: str) -> None:
                progress.update(task_id, total=total, completed=current, description=f"Processing {station}")

            metadata_df = compute_all_metadata(
                year_filter=year_filter,
                station_filter=station_filter,
                progress_callback=progress_callback,
                max_workers=workers,
            )

        if len(metadata_df) == 0:
            print_warning("No metadata computed. Check that parquet files exist.")
            raise typer.Exit(1)

        # Enrich with NOAA station inventory (lat/lon/elevation, WMO ID, ICAO, etc.)
        console.print("\n[bold]Enriching metadata from NOAA station inventory...[/bold]")
        metadata_df = enrich_metadata_from_station_list(metadata_df)

        # Save metadata
        save_metadata(metadata_df, cleaned=False, export_csv=export_csv)

        # Print summary
        no_hourly_count = (metadata_df["total_observation_count"] == 0).sum()
        print_success(f"Computed metadata for {len(metadata_df)} stations")
        print_summary_table("Metadata Summary", {
            "Total stations": len(metadata_df),
            "Stations with hourly data": len(metadata_df) - no_hourly_count,
            "Stations without hourly data": no_hourly_count,
            "Output file": str(METADATA_PARQUET),
            "CSV export": str(export_csv),
        })


@app.command()
def show(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of rows to show"),
    cleaned: bool = typer.Option(False, "--cleaned", "-c", help="Show cleaned metadata"),
) -> None:
    """Show computed metadata."""
    metadata = load_metadata(cleaned=cleaned)

    if metadata is None:
        print_warning("No metadata found. Run 'compute' first.")
        raise typer.Exit(1)

    # Create table
    table = Table(title=f"Station Metadata ({'cleaned' if cleaned else 'raw'})")
    table.add_column("Station ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Lat", justify="right")
    table.add_column("Lon", justify="right")
    table.add_column("Years", justify="right")
    table.add_column("Obs Count", justify="right")
    table.add_column("Temp %", justify="right", style="green")

    for row in metadata.head(limit).iter_rows(named=True):
        years = row.get("years_available", [])
        # Parse JSON string if needed
        if isinstance(years, str):
            try:
                years = json.loads(years)
            except (json.JSONDecodeError, TypeError):
                years = []
        year_count = len(years) if isinstance(years, list) else 0

        table.add_row(
            row.get("station_id", ""),
            (row.get("station_name", "") or "")[:30],
            f"{row.get('latitude', 0):.2f}" if row.get("latitude") else "",
            f"{row.get('longitude', 0):.2f}" if row.get("longitude") else "",
            str(year_count),
            f"{row.get('total_observation_count', 0):,}",
            f"{row.get('temperature_completeness_pct', 0):.1f}%",
        )

    console.print(table)
    console.print(f"\n[dim]Showing {min(limit, len(metadata))} of {len(metadata)} stations[/dim]")


@app.command()
def stats() -> None:
    """Show metadata statistics."""
    metadata = load_metadata(cleaned=False)

    if metadata is None:
        print_warning("No metadata found. Run 'compute' first.")
        raise typer.Exit(1)

    # Compute statistics
    stats_dict = {
        "Total stations": len(metadata),
        "Avg observations per station": f"{metadata['total_observation_count'].mean():,.0f}",
        "Total observations": f"{metadata['total_observation_count'].sum():,}",
        "Avg temperature completeness": f"{metadata['temperature_completeness_pct'].mean():.1f}%",
        "Stations with >90% temp coverage": f"{(metadata['temperature_completeness_pct'] > 90).sum()}",
    }

    print_summary_table("Metadata Statistics", stats_dict)


if __name__ == "__main__":
    app()
