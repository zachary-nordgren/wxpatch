# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "polars",
#     "plotly",
#     "torch",
#     "numpy",
#     "pandas",
#     "pyarrow==23.0.0",
# ]
# ///
"""
Gap Analysis Notebook

Interactive exploration of missing data patterns in GHCNh weather data.
Analyzes natural gap distributions and compares with synthetic masking strategies.

Run with: uv run marimo edit notebooks/02_gap_analysis.py
"""

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def imports():
    """Import required libraries."""
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import torch
    from plotly.subplots import make_subplots

    # Add project to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))

    from weather_imputation.config.paths import GHCNH_RAW_DIR, PROCESSED_DIR
    from weather_imputation.data.ghcnh_loader import (
        TIER1_VARIABLES,
        extract_tier1_variables,
        load_station_data,
    )
    from weather_imputation.data.masking import (
        apply_mar_mask,
        apply_mcar_mask,
        apply_mnar_mask,
        apply_realistic_mask,
    )
    from weather_imputation.data.metadata import load_metadata

    return (
        GHCNH_RAW_DIR,
        PROCESSED_DIR,
        Path,
        TIER1_VARIABLES,
        apply_mar_mask,
        apply_mcar_mask,
        apply_mnar_mask,
        apply_realistic_mask,
        extract_tier1_variables,
        go,
        load_metadata,
        load_station_data,
        make_subplots,
        mo,
        np,
        pl,
        px,
        torch,
    )


@app.cell
def header(mo):
    """Display notebook header."""
    mo.md("""
    # Gap Analysis

    Interactive exploration of missing data patterns in GHCNh weather data.

    **Objectives:**
    1. Analyze natural gap patterns in real weather data
    2. Characterize gap length distributions
    3. Compare synthetic masking strategies (MCAR, MAR, MNAR, Realistic)
    4. Visualize gap characteristics per variable

    Use this notebook to understand missingness patterns and validate
    synthetic masking strategies against real-world data.

    ---

    ## 1. Data Loading
    """)
    return


@app.cell
def load_station_list(PROCESSED_DIR, load_metadata, mo, pl):
    """Load station metadata and select sample stations."""
    metadata = load_metadata(cleaned=True)

    if metadata is None:
        mo.stop(
            True,
            mo.md(
                "**Error**: No metadata file found. "
                "Run `compute_metadata.py` and `clean_metadata.py` first."
            ),
        )

    # Filter to high-quality stations for analysis
    high_quality = metadata.filter(
        (pl.col("temperature_completeness_pct") >= 80)
        & (pl.col("years_available") >= 10)
        & (pl.col("valid_coordinates") == True)  # noqa: E712
    )

    mo.md(f"""
    Loaded metadata for {len(metadata):,} stations.
    Found {len(high_quality):,} high-quality stations (≥80% temp completeness, ≥10 years).

    We'll sample a few stations for detailed gap analysis.
    """)

    # Sample stations from different geographic regions
    # Get one station from each of several US regions
    sampled_stations = (
        high_quality.filter(pl.col("country_code") == "US")
        .sort("temperature_completeness_pct", descending=True)
        .head(10)
    )

    return high_quality, metadata, sampled_stations


@app.cell
def station_selector(mo, sampled_stations):
    """UI for selecting a station to analyze."""
    mo.md("## 2. Station Selection")

    # Create dropdown options
    station_options = {
        (
            f"{row['station_id']} - {row['station_name']} "
            f"({row['temperature_completeness_pct']:.1f}% complete)"
        ): row["station_id"]
        for row in sampled_stations.iter_rows(named=True)
    }

    selected_station_dropdown = mo.ui.dropdown(
        options=station_options,
        value=list(station_options.values())[0],
        label="Select a station to analyze:",
    )

    return selected_station_dropdown, station_options


@app.cell
def show_station_selector(mo, selected_station_dropdown):
    """Display station selector."""
    mo.md("Select a station to analyze its gap patterns:")
    return mo.hstack([selected_station_dropdown])


@app.cell
def load_station_time_series(
    extract_tier1_variables, load_station_data, mo, selected_station_dropdown
):
    """Load time series data for selected station."""
    station_id = selected_station_dropdown.value

    if station_id is None:
        mo.stop(True, mo.md("Select a station to continue."))

    mo.md(f"Loading data for station: **{station_id}**")

    # Load data for recent years (2018-2023 for faster loading)
    try:
        df_full = load_station_data(
            station_id=station_id,
            years=list(range(2018, 2024)),
            report_types=["AUTO", "FM-15"],
        )

        # Extract Tier 1 variables
        df = extract_tier1_variables(df_full)

        # Get temperature data for gap analysis
        if "temperature" not in df.columns:
            mo.stop(True, mo.md(f"**Error**: Station {station_id} has no temperature data."))

        available_vars = [
            col
            for col in df.columns
            if col
            in [
                "temperature",
                "dew_point_temperature",
                "sea_level_pressure",
                "wind_speed",
                "wind_direction",
                "relative_humidity",
            ]
        ]

        mo.md(f"""
        Loaded {len(df):,} observations from 2018-2023.

        Variables available: {', '.join(available_vars)}
        """)

        df_loaded = df

    except Exception as e:
        mo.stop(True, mo.md(f"**Error loading data**: {str(e)}"))

    return df_full, df_loaded, station_id


@app.cell
def analyze_natural_gaps(df_loaded, mo, np, pl):
    """Analyze natural gap patterns in the loaded data."""
    mo.md("## 3. Natural Gap Analysis")

    # Analyze temperature gaps (most commonly available variable)
    temp_data = df_loaded.select(["DATE", "temperature"]).sort("DATE")

    # Identify gaps (null values)
    natural_gaps = []
    _current_gap_start = None
    _current_gap_length = 0

    for _i, _row in enumerate(temp_data.iter_rows(named=True)):
        _is_missing = _row["temperature"] is None

        if _is_missing:
            if _current_gap_start is None:
                _current_gap_start = _i
            _current_gap_length += 1
        else:
            if _current_gap_start is not None:
                # Gap ended, record it
                natural_gaps.append(
                    {
                        "start_idx": _current_gap_start,
                        "length_hours": _current_gap_length,
                    }
                )
                _current_gap_start = None
                _current_gap_length = 0

    # Handle gap at end of series
    if _current_gap_start is not None:
        natural_gaps.append(
            {"start_idx": _current_gap_start, "length_hours": _current_gap_length}
        )

    gaps_df = pl.DataFrame(natural_gaps) if natural_gaps else pl.DataFrame({"length_hours": []})

    # Calculate gap statistics
    _total_obs = len(temp_data)
    _total_missing = temp_data.select(pl.col("temperature").is_null().sum()).item()
    _missing_pct = (_total_missing / _total_obs) * 100 if _total_obs > 0 else 0

    gap_stats = {
        "total_gaps": len(natural_gaps),
        "total_missing_hours": _total_missing,
        "missing_percentage": _missing_pct,
        "mean_gap_length": (
            float(np.mean([g["length_hours"] for g in natural_gaps]))
            if natural_gaps
            else 0
        ),
        "median_gap_length": (
            float(np.median([g["length_hours"] for g in natural_gaps]))
            if natural_gaps
            else 0
        ),
        "max_gap_length": (
            max([g["length_hours"] for g in natural_gaps]) if natural_gaps else 0
        ),
    }

    return gap_stats, gaps_df, temp_data


@app.cell
def display_gap_stats(gap_stats, mo):
    """Display gap statistics."""
    max_gap_days = gap_stats["max_gap_length"] / 24
    mo.md(f"""
    ### Gap Statistics

    - **Total gaps:** {gap_stats['total_gaps']:,}
    - **Total missing hours:** {gap_stats['total_missing_hours']:,}
    - **Missing percentage:** {gap_stats['missing_percentage']:.2f}%
    - **Mean gap length:** {gap_stats['mean_gap_length']:.1f} hours
    - **Median gap length:** {gap_stats['median_gap_length']:.1f} hours
    - **Max gap length:** {gap_stats['max_gap_length']:,} hours ({max_gap_days:.1f} days)
    """)
    return


@app.cell
def plot_gap_distribution(gaps_df, mo, np, px):
    """Plot gap length distribution."""
    if len(gaps_df) == 0:
        mo.md("**No gaps found in this station's data.**")
        gap_dist_fig = None
    else:
        mo.md("### Gap Length Distribution")

        # Create histogram with log scale
        gap_dist_fig = px.histogram(
            gaps_df.to_pandas(),
            x="length_hours",
            nbins=50,
            title="Distribution of Gap Lengths (Hours)",
            labels={"length_hours": "Gap Length (hours)", "count": "Frequency"},
            log_y=True,
        )

        gap_dist_fig.update_layout(
            xaxis_title="Gap Length (hours)",
            yaxis_title="Frequency (log scale)",
            showlegend=False,
        )

        # Add vertical lines for reference durations
        gap_dist_fig.add_vline(
            x=6, line_dash="dash", line_color="red", annotation_text="6h (short)"
        )
        gap_dist_fig.add_vline(
            x=72, line_dash="dash", line_color="orange", annotation_text="72h (medium)"
        )

    return (gap_dist_fig,)


@app.cell
def show_gap_distribution(gap_dist_fig, mo):
    """Display gap distribution plot."""
    _display = mo.ui.plotly(gap_dist_fig) if gap_dist_fig is not None else None
    return


@app.cell
def categorize_gaps(gaps_df, mo, pl):
    """Categorize gaps into short/medium/long."""
    if len(gaps_df) == 0:
        _msg = mo.md("_No gaps to categorize._")
        gap_categories = None
        category_counts = None
    else:
        _msg = mo.md("### Gap Categorization")

        # Categorize gaps
        gap_categories = gaps_df.with_columns(
            pl.when(pl.col("length_hours") <= 6)
            .then(pl.lit("short (≤6h)"))
            .when(pl.col("length_hours") <= 72)
            .then(pl.lit("medium (6-72h)"))
            .otherwise(pl.lit("long (>72h)"))
            .alias("category")
        )

        # Count by category
        category_counts = (
            gap_categories.group_by("category")
            .agg(
                [
                    pl.count().alias("count"),
                    pl.col("length_hours").sum().alias("total_hours"),
                ]
            )
            .sort("count", descending=True)
        )

        _msg = mo.md(f"""
        **Gap Categories:**

        {category_counts.to_pandas().to_markdown(index=False)}

        This distribution informs our realistic masking strategy:
        - Short gaps (1-6h): Sensor noise, brief outages
        - Medium gaps (6-72h): Maintenance, temporary failures
        - Long gaps (>72h): Equipment failures, extended outages
        """)

    return category_counts, gap_categories


@app.cell
def synthetic_masking_controls(mo):
    """UI controls for synthetic masking parameters."""
    mo.md("## 4. Synthetic Masking Comparison")

    mo.md("""
    Now let's compare synthetic masking strategies with the natural gap patterns.

    Select masking parameters below:
    """)

    missing_ratio_slider = mo.ui.slider(
        start=0.05, stop=0.5, value=0.2, step=0.05, label="Missing ratio"
    )

    min_gap_slider = mo.ui.slider(
        start=1, stop=12, value=1, step=1, label="Min gap length (hours)"
    )

    max_gap_slider = mo.ui.slider(
        start=12, stop=336, value=168, step=12, label="Max gap length (hours)"
    )

    return max_gap_slider, min_gap_slider, missing_ratio_slider


@app.cell
def show_masking_controls(max_gap_slider, min_gap_slider, missing_ratio_slider, mo):
    """Display masking controls."""
    return mo.vstack(
        [
            mo.md("**Masking Parameters:**"),
            missing_ratio_slider,
            min_gap_slider,
            max_gap_slider,
        ]
    )


@app.cell
def generate_synthetic_masks(
    apply_mar_mask,
    apply_mcar_mask,
    apply_mnar_mask,
    apply_realistic_mask,
    max_gap_slider,
    min_gap_slider,
    missing_ratio_slider,
    mo,
    np,
    temp_data,
    torch,
):
    """Generate synthetic masks using different strategies."""
    mo.md("Generating synthetic masks...")

    # Create synthetic data tensor (use temperature values)
    # Shape: (1, T, 1) for single sample, T timesteps, 1 variable
    temp_values = temp_data.select("temperature").to_numpy().flatten()

    # Replace nulls with mean for masking (we'll mask them anyway)
    mean_temp = temp_values[~np.isnan(temp_values)].mean()
    temp_values = np.where(np.isnan(temp_values), mean_temp, temp_values)

    data_tensor = torch.from_numpy(temp_values).float().reshape(1, -1, 1)

    # Generate masks with different strategies
    seed = 42  # For reproducibility

    mask_mcar = apply_mcar_mask(
        data_tensor,
        missing_ratio=missing_ratio_slider.value,
        min_gap_length=min_gap_slider.value,
        max_gap_length=max_gap_slider.value,
        seed=seed,
    )

    mask_mar = apply_mar_mask(
        data_tensor,
        missing_ratio=missing_ratio_slider.value,
        min_gap_length=min_gap_slider.value,
        max_gap_length=max_gap_slider.value,
        condition_variable=0,
        seed=seed,
    )

    mask_mnar = apply_mnar_mask(
        data_tensor,
        missing_ratio=missing_ratio_slider.value,
        min_gap_length=min_gap_slider.value,
        max_gap_length=max_gap_slider.value,
        target_variable=0,
        seed=seed,
    )

    mask_realistic = apply_realistic_mask(
        data_tensor, missing_ratio=missing_ratio_slider.value, seed=seed
    )

    masks = {
        "MCAR": mask_mcar,
        "MAR": mask_mar,
        "MNAR": mask_mnar,
        "Realistic": mask_realistic,
    }

    return data_tensor, mask_mar, mask_mcar, mask_mnar, mask_realistic, masks, mean_temp, seed


@app.cell
def analyze_synthetic_gaps(masks, mo, np, pl):
    """Analyze gap patterns in synthetic masks."""
    mo.md("### Synthetic Gap Statistics")

    synthetic_stats = []

    for _strategy_name, _mask in masks.items():
        # Convert mask to numpy (shape: 1, T, 1)
        _mask_np = _mask.squeeze().numpy()  # Shape: (T,)

        # Find gaps
        _synthetic_gaps = []
        _syn_gap_length = 0

        for _j in range(len(_mask_np)):
            _syn_is_missing = not _mask_np[_j]

            if _syn_is_missing:
                _syn_gap_length += 1
            else:
                if _syn_gap_length > 0:
                    _synthetic_gaps.append(_syn_gap_length)
                    _syn_gap_length = 0

        # Handle gap at end
        if _syn_gap_length > 0:
            _synthetic_gaps.append(_syn_gap_length)

        # Calculate statistics
        _syn_total_missing = (~_mask).sum().item()
        _syn_total_obs = _mask.numel()
        _syn_missing_pct = (_syn_total_missing / _syn_total_obs) * 100

        synthetic_stats.append(
            {
                "strategy": _strategy_name,
                "total_gaps": len(_synthetic_gaps),
                "missing_pct": _syn_missing_pct,
                "mean_gap": float(np.mean(_synthetic_gaps)) if _synthetic_gaps else 0,
                "median_gap": float(np.median(_synthetic_gaps)) if _synthetic_gaps else 0,
                "max_gap": max(_synthetic_gaps) if _synthetic_gaps else 0,
                "gaps": _synthetic_gaps,
            }
        )

    synthetic_stats_df = pl.DataFrame(
        [
            {
                "Strategy": s["strategy"],
                "Total Gaps": s["total_gaps"],
                "Missing %": f"{s['missing_pct']:.1f}",
                "Mean Gap": f"{s['mean_gap']:.1f}h",
                "Median Gap": f"{s['median_gap']:.1f}h",
                "Max Gap": f"{s['max_gap']}h",
            }
            for s in synthetic_stats
        ]
    )

    return synthetic_stats, synthetic_stats_df


@app.cell
def display_synthetic_stats(mo, synthetic_stats_df):
    """Display synthetic gap statistics table."""
    mo.md(f"""
    {synthetic_stats_df.to_pandas().to_markdown(index=False)}
    """)
    return


@app.cell
def plot_synthetic_comparison(gap_stats, go, make_subplots, mo, np, synthetic_stats):
    """Plot comparison of natural vs synthetic gap distributions."""
    mo.md("### Gap Length Distribution Comparison")

    # Create subplots
    comparison_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("MCAR", "MAR", "MNAR", "Realistic"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Define bin edges (log scale)
    bin_edges = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    for _idx, _stat in enumerate(synthetic_stats):
        _plot_row = (_idx // 2) + 1
        _plot_col = (_idx % 2) + 1

        _plot_gaps = _stat["gaps"]

        if _plot_gaps:
            # Compute histogram
            _hist, _ = np.histogram(_plot_gaps, bins=bin_edges)

            comparison_fig.add_trace(
                go.Bar(
                    x=[f"{bin_edges[_k]}-{bin_edges[_k+1]}" for _k in range(len(_hist))],
                    y=_hist,
                    name=_stat["strategy"],
                    showlegend=False,
                ),
                row=_plot_row,
                col=_plot_col,
            )

    comparison_fig.update_xaxes(title_text="Gap Length (hours)", type="category")
    comparison_fig.update_yaxes(title_text="Frequency")
    comparison_fig.update_layout(height=600, title_text="Synthetic Gap Distributions by Strategy")

    return comparison_fig


@app.cell
def show_synthetic_comparison(comparison_fig, mo):
    """Display synthetic comparison plot."""
    return mo.ui.plotly(comparison_fig)


@app.cell
def comparison_summary(mo):
    """Summary and insights."""
    mo.md("""
    ## 5. Summary and Insights

    ### Key Observations:

    1. **Natural Gaps**: Real-world gaps tend to cluster at specific durations:
       - Very short (1-2 hours): Sensor noise, brief communication issues
       - Medium (6-24 hours): Scheduled maintenance windows
       - Long (multiple days): Equipment failures requiring replacement

    2. **MCAR (Missing Completely At Random)**: Generates gaps uniformly across the series.
       - Good for: Baseline evaluation, theoretical analysis
       - Limitation: Doesn't reflect real-world patterns

    3. **MAR (Missing At Random)**: More gaps during extreme weather
       (conditional on observed values).
       - Good for: Testing imputation under biased observation conditions
       - Reflects: Sensor failures correlated with extreme observed conditions

    4. **MNAR (Missing Not At Random)**: Missing values themselves are extreme.
       - Good for: Most challenging imputation scenario
       - Reflects: Sensor failures caused by the extreme values being measured

    5. **Realistic**: Mimics observed gap length distribution from GHCNh data.
       - Good for: Most realistic evaluation scenario
       - Based on: Empirical analysis of natural gap patterns

    ### Recommendations:

    - **For training**: Use realistic masking to simulate real-world conditions
    - **For evaluation**: Test all four strategies to understand model robustness
    - **For ablation studies**: MCAR provides clean baseline without confounding factors
    - **For stress testing**: MNAR challenges models with hardest imputation scenarios

    ---

    **Next Steps:**
    - Explore multiple stations to validate gap pattern consistency
    - Analyze per-variable gap patterns (temperature vs pressure vs humidity)
    - Use these insights to refine masking strategies for model training
    """)
    return


if __name__ == "__main__":
    app.run()
