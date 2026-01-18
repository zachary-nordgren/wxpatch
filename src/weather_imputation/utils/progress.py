"""Rich-based progress display utilities."""

from contextlib import contextmanager
from typing import Any, Generator

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    DownloadColumn,
    TransferSpeedColumn,
)
from rich.table import Table

console = Console()


def create_download_progress() -> Progress:
    """Progress bar for file downloads with speed and ETA."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def create_processing_progress() -> Progress:
    """Progress bar for batch processing operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def create_simple_progress() -> Progress:
    """Simple progress bar with percentage and time."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        "•",
        TimeElapsedColumn(),
        console=console,
    )


@contextmanager
def status_spinner(message: str) -> Generator[None, None, None]:
    """Simple spinner for indeterminate operations.

    Usage:
        with status_spinner("Loading data..."):
            do_something()
    """
    with console.status(message, spinner="dots"):
        yield


def print_summary_table(title: str, data: dict[str, Any]) -> None:
    """Print a formatted summary table.

    Args:
        title: Table title
        data: Dictionary of metric names to values
    """
    table = Table(title=title)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    for key, value in data.items():
        table.add_row(key, str(value))
    console.print(table)


def print_error(message: str) -> None:
    """Print an error message in red."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_success(message: str) -> None:
    """Print a success message in green."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_warning(message: str) -> None:
    """Print a warning message in yellow."""
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")
