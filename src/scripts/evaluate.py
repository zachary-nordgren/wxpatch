#!/usr/bin/env python3
"""CLI for evaluating imputation model performance.

This script loads a trained model (or creates a classical baseline), runs inference
on test data, computes evaluation metrics, and saves results.

Usage:
    python evaluate.py --model linear                          # Evaluate linear baseline
    python evaluate.py --model mice --config configs/model/mice.yaml
    python evaluate.py --checkpoint checkpoints/saits_best.pt  # Evaluate trained model
    python evaluate.py --model linear --stratified             # Include stratified analysis
"""

import json
import logging
from pathlib import Path
from typing import Any

import polars as pl
import torch
import typer
from rich.console import Console
from rich.table import Table

from weather_imputation.config.paths import PROCESSED_DIR, RESULTS_DIR
from weather_imputation.evaluation.metrics import compute_all_metrics
from weather_imputation.evaluation.stratified import (
    stratify_by_gap_length,
    stratify_by_season,
    stratify_by_variable,
)
from weather_imputation.models.classical.linear import LinearInterpolationImputer
from weather_imputation.models.classical.mice import MICEImputer
from weather_imputation.models.classical.spline import AkimaSplineImputer
from weather_imputation.utils.progress import (
    print_error,
    print_info,
    print_success,
)

app = typer.Typer(help="Evaluate imputation model performance on test data.")
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


def load_test_data(
    test_file: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load test data from parquet file.

    Args:
        test_file: Path to test.parquet file

    Returns:
        Tuple of (observed, mask, target, timestamps) tensors
    """
    print_info(f"Loading test data from: {test_file}")

    # Load parquet file
    df = pl.read_parquet(test_file)

    # Extract variable columns (excluding metadata columns)
    metadata_cols = {"DATE", "station_id", "latitude", "longitude", "elevation"}
    variable_cols = [col for col in df.columns if col not in metadata_cols]

    # Convert to tensors
    # For simplicity, treat entire dataset as one batch
    # Shape will be (1, T, V) where T is number of rows, V is number of variables
    data = df.select(variable_cols).to_numpy()
    observed = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    # For test data, we assume all values are observed initially
    # Synthetic masks will be applied during evaluation
    mask = ~torch.isnan(observed)
    target = observed.clone()

    # Extract timestamps
    timestamps = torch.tensor(
        df.select("DATE").to_series().to_numpy().astype("int64") // 1_000_000_000
    ).unsqueeze(0)

    print_success(
        f"Loaded test data: {observed.shape[1]} timesteps, "
        f"{observed.shape[2]} variables"
    )

    return observed, mask, target, timestamps


def create_classical_model(
    model_name: str, config_path: Path | None = None
) -> Any:
    """Create a classical imputation model.

    Args:
        model_name: Name of model (linear, spline, mice)
        config_path: Optional path to config file with hyperparameters

    Returns:
        Initialized imputer instance
    """
    # Load config if provided
    config = {}
    if config_path and config_path.exists():
        if config_path.suffix == ".yaml":
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            with open(config_path) as f:
                config = json.load(f)

    # Create model based on name
    if model_name == "linear":
        max_gap_length = config.get("max_gap_length")
        return LinearInterpolationImputer(max_gap_length=max_gap_length)
    elif model_name == "spline":
        max_gap_length = config.get("max_gap_length")
        return AkimaSplineImputer(max_gap_length=max_gap_length)
    elif model_name == "mice":
        predictor = config.get("predictor", "bayesian_ridge")
        max_iter = config.get("max_iter", 10)
        n_imputations = config.get("n_imputations", 5)
        return MICEImputer(
            predictor=predictor, max_iter=max_iter, n_imputations=n_imputations
        )
    else:
        raise ValueError(
            f"Unknown classical model: {model_name}. "
            "Supported: linear, spline, mice"
        )


def load_trained_model(checkpoint_path: Path) -> Any:
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)

    Returns:
        Loaded model ready for inference
    """
    print_info(f"Loading model from checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Determine model type from checkpoint metadata
    model_type = checkpoint.get("model_type", "unknown")

    if model_type in ["saits", "csdi"]:
        # These will be implemented in later tasks
        raise NotImplementedError(
            f"Loading {model_type} models not yet implemented. "
            "Use classical models (linear, spline, mice) for now."
        )
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")


def apply_synthetic_mask(
    observed: torch.Tensor,
    mask: torch.Tensor,
    masking_strategy: str = "realistic",
    mask_ratio: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply synthetic masking to create evaluation gaps.

    Args:
        observed: Full observed data (N, T, V)
        mask: Original observation mask (N, T, V)
        masking_strategy: Type of masking (mcar, mar, mnar, realistic)
        mask_ratio: Fraction of values to mask

    Returns:
        Tuple of (masked_observed, eval_mask, gap_lengths)
        - masked_observed: Data with synthetic gaps (N, T, V)
        - eval_mask: Mask indicating where to evaluate (N, T, V)
        - gap_lengths: Length of each gap in timesteps (N, T, V)
    """
    from weather_imputation.data.masking import (
        apply_mar_mask,
        apply_mcar_mask,
        apply_mnar_mask,
        apply_realistic_mask,
    )

    # Apply synthetic masking
    if masking_strategy == "mcar":
        synthetic_mask = apply_mcar_mask(
            observed, missing_ratio=mask_ratio, min_gap_length=1, max_gap_length=168
        )
    elif masking_strategy == "mar":
        synthetic_mask = apply_mar_mask(
            observed, missing_ratio=mask_ratio, extreme_percentile=0.15
        )
    elif masking_strategy == "mnar":
        synthetic_mask = apply_mnar_mask(
            observed, missing_ratio=mask_ratio, extreme_multiplier=5.0
        )
    elif masking_strategy == "realistic":
        synthetic_mask = apply_realistic_mask(
            observed,
            missing_ratio=mask_ratio,
            gap_distribution="empirical",
        )
    else:
        raise ValueError(f"Unknown masking strategy: {masking_strategy}")

    # Create masked observed data (set synthetic gaps to 0)
    masked_observed = observed.clone()
    masked_observed[~synthetic_mask] = 0.0

    # Evaluation mask: only evaluate where we synthetically masked
    eval_mask = mask & ~synthetic_mask

    # Compute gap lengths for stratification
    gap_lengths = compute_gap_lengths(synthetic_mask)

    return masked_observed, eval_mask, gap_lengths


def compute_gap_lengths(mask: torch.Tensor) -> torch.Tensor:
    """Compute the length of the gap each missing position belongs to.

    Args:
        mask: Observation mask (N, T, V), True=observed

    Returns:
        Gap lengths tensor (N, T, V), each missing position labeled with gap length
    """
    N, T, V = mask.shape
    gap_lengths = torch.zeros_like(mask, dtype=torch.float32)

    # Process each sample and variable independently
    for n in range(N):
        for v in range(V):
            obs = mask[n, :, v]
            gaps = gap_lengths[n, :, v]

            # Find gap starts and ends
            i = 0
            while i < T:
                if not obs[i]:  # Start of gap
                    gap_start = i
                    gap_end = i
                    # Find gap end
                    while gap_end < T and not obs[gap_end]:
                        gap_end += 1
                    # Label all positions in this gap with gap length
                    gap_len = gap_end - gap_start
                    gaps[gap_start:gap_end] = gap_len
                    i = gap_end
                else:
                    i += 1

    return gap_lengths


def save_results(
    results: dict[str, Any],
    output_file: Path,
    format: str = "json",
) -> None:
    """Save evaluation results to file.

    Args:
        results: Dictionary of results to save
        output_file: Output file path
        format: Output format (json or parquet)
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print_success(f"Results saved to: {output_file}")

    elif format == "parquet":
        # Flatten nested results for parquet
        flat_results = []
        for key, value in results.items():
            if isinstance(value, dict):
                row = {"metric_group": key}
                row.update(value)
                flat_results.append(row)
            else:
                flat_results.append({"metric": key, "value": value})

        df = pl.DataFrame(flat_results)
        df.write_parquet(output_file)
        print_success(f"Results saved to: {output_file}")

    else:
        raise ValueError(f"Unknown format: {format}")


def display_results(results: dict[str, Any], stratified: bool = False) -> None:
    """Display results in a formatted table.

    Args:
        results: Dictionary of results
        stratified: Whether to include stratified results
    """
    # Overall metrics table
    if "overall" in results:
        table = Table(title="Overall Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        for metric, value in results["overall"].items():
            table.add_row(metric.upper(), f"{value:.4f}")

        console.print(table)

    # Stratified results
    if stratified and "stratified" in results:
        console.print("\n[bold]Stratified Results:[/bold]")

        # Gap length stratification
        if "gap_length" in results["stratified"]:
            table = Table(title="By Gap Length")
            table.add_column("Gap Range", style="cyan")
            table.add_column("RMSE", style="magenta")
            table.add_column("MAE", style="magenta")
            table.add_column("R²", style="magenta")

            for gap_range, metrics in results["stratified"]["gap_length"].items():
                table.add_row(
                    gap_range,
                    f"{metrics['rmse']:.4f}",
                    f"{metrics['mae']:.4f}",
                    f"{metrics['r2']:.4f}",
                )

            console.print(table)

        # Variable stratification
        if "variable" in results["stratified"]:
            table = Table(title="By Variable")
            table.add_column("Variable", style="cyan")
            table.add_column("RMSE", style="magenta")
            table.add_column("MAE", style="magenta")
            table.add_column("R²", style="magenta")

            for var_name, metrics in results["stratified"]["variable"].items():
                table.add_row(
                    var_name,
                    f"{metrics['rmse']:.4f}",
                    f"{metrics['mae']:.4f}",
                    f"{metrics['r2']:.4f}",
                )

            console.print(table)


@app.command()
def evaluate(
    test_file: Path = typer.Option(
        PROCESSED_DIR / "test.parquet",
        "--test-file",
        "-t",
        help="Path to test data parquet file",
    ),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Classical model name (linear, spline, mice)",
    ),
    checkpoint: Path = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Path to trained model checkpoint (.pt file)",
    ),
    config: Path = typer.Option(
        None,
        "--config",
        help="Path to model config file (YAML or JSON)",
    ),
    output_dir: Path = typer.Option(
        RESULTS_DIR,
        "--output-dir",
        "-o",
        help="Output directory for results",
    ),
    masking_strategy: str = typer.Option(
        "realistic",
        "--masking-strategy",
        help="Synthetic masking strategy (mcar, mar, mnar, realistic)",
    ),
    mask_ratio: float = typer.Option(
        0.2,
        "--mask-ratio",
        min=0.0,
        max=1.0,
        help="Fraction of values to mask for evaluation",
    ),
    stratified: bool = typer.Option(
        False,
        "--stratified",
        help="Include stratified analysis (gap length, season, variable)",
    ),
    variable_names: str = typer.Option(
        "temperature,dew_point_temperature,sea_level_pressure,wind_speed,wind_direction,relative_humidity",
        "--variable-names",
        help="Comma-separated variable names for stratification",
    ),
    output_format: str = typer.Option(
        "json",
        "--format",
        help="Output format (json or parquet)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-error output",
    ),
) -> None:
    """Evaluate imputation model on test data.

    Either --model or --checkpoint must be specified.
    """
    setup_logging(verbose, quiet)

    # Validate inputs
    if model is None and checkpoint is None:
        print_error("Either --model or --checkpoint must be specified")
        raise typer.Exit(1)

    if model is not None and checkpoint is not None:
        print_error("Cannot specify both --model and --checkpoint")
        raise typer.Exit(1)

    if not test_file.exists():
        print_error(f"Test file not found: {test_file}")
        raise typer.Exit(1)

    # Load model
    if model:
        print_info(f"Creating classical model: {model}")
        imputer = create_classical_model(model, config)
        # Classical models don't need fitting
        imputer.fit(None, None)  # type: ignore
        model_name = model
    else:
        imputer = load_trained_model(checkpoint)  # type: ignore
        model_name = checkpoint.stem  # type: ignore

    # Load test data
    observed, mask, target, timestamps = load_test_data(test_file)

    # Apply synthetic masking
    print_info(f"Applying synthetic masking: {masking_strategy} ({mask_ratio:.0%})")
    masked_observed, eval_mask, gap_lengths = apply_synthetic_mask(
        observed, mask, masking_strategy, mask_ratio
    )

    print_info(f"Evaluation positions: {eval_mask.sum().item()}")

    # Run imputation
    print_info("Running imputation...")
    with torch.no_grad():
        imputed = imputer.impute(masked_observed, eval_mask)

    # Compute overall metrics
    print_info("Computing metrics...")
    overall_metrics = compute_all_metrics(target, imputed, eval_mask)

    results = {
        "model": model_name,
        "masking_strategy": masking_strategy,
        "mask_ratio": mask_ratio,
        "n_eval_positions": eval_mask.sum().item(),
        "overall": overall_metrics,
    }

    # Compute stratified metrics if requested
    if stratified:
        print_info("Computing stratified metrics...")
        var_names = variable_names.split(",")

        stratified_results = {}

        # Gap length stratification
        gap_length_results = stratify_by_gap_length(
            target, imputed, eval_mask, gap_lengths
        )
        stratified_results["gap_length"] = gap_length_results

        # Variable stratification
        variable_results = stratify_by_variable(
            target, imputed, eval_mask, variable_names=var_names
        )
        stratified_results["variable"] = variable_results

        # Seasonal stratification
        season_results = stratify_by_season(target, imputed, eval_mask, timestamps)
        stratified_results["season"] = season_results

        results["stratified"] = stratified_results

    # Display results
    display_results(results, stratified)

    # Save results
    output_file = output_dir / f"{model_name}_results.{output_format}"
    save_results(results, output_file, output_format)

    print_success("Evaluation complete!")


if __name__ == "__main__":
    app()
