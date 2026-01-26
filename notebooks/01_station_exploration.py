# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "polars",
#     "plotly",
#     "folium",
#     "scikit-learn",
#     "pandas",
#     "pyarrow==23.0.0",
# ]
# ///
"""
Station Exploration Notebook

Interactive exploration of GHCNh station metadata for filtering and clustering
stations by completeness criteria.

Run with: uv run marimo edit notebooks/01_station_exploration.py
"""

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def imports():
    """Import required libraries."""
    import json

    # Project imports - add src to path
    import sys
    from pathlib import Path

    import folium
    import marimo as mo
    import plotly.express as px
    import polars as pl
    from folium.plugins import MarkerCluster
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))

    from weather_imputation.config.paths import PROCESSED_DIR
    from weather_imputation.data.metadata import load_metadata
    return (
        KMeans,
        MarkerCluster,
        PROCESSED_DIR,
        StandardScaler,
        folium,
        json,
        load_metadata,
        mo,
        pl,
        px,
    )


@app.cell
def load_data(load_metadata, mo):
    """Load metadata and display header."""
    mo.md("""
    # Station Exploration

    Interactive exploration of GHCNh station metadata for:
    - Filtering stations by completeness criteria
    - Visualizing geographic coverage
    - Clustering stations by characteristics

    Use the sliders below to adjust filtering thresholds and see the
    resulting station count and geographic distribution.

    ---
    ## 1. Data Loading
    """)

    # Try cleaned first, fall back to raw
    metadata = load_metadata(cleaned=True)

    if metadata is None:
        metadata = load_metadata(cleaned=False)

    if metadata is None:
        mo.stop(
            True,
            mo.md("**Error**: No metadata file found. Run `compute_metadata.py` first.")
        )

    return (metadata,)


@app.cell
def filtering_controls(mo):
    """Interactive threshold controls for filtering."""
    temp_completeness = mo.ui.slider(
        start=0, stop=100, value=70, step=5,
        label="Min Temperature Completeness (%)"
    )

    min_years = mo.ui.slider(
        start=1, stop=30, value=5, step=1,
        label="Min Time Span (years)"
    )

    min_observations = mo.ui.slider(
        start=0, stop=100000, value=20000, step=5000,
        label="Min Observations"
    )

    require_valid_coords = mo.ui.checkbox(
        value=True,
        label="Require Valid Coordinates"
    )
    return min_observations, min_years, require_valid_coords, temp_completeness


@app.cell
def apply_filters(
    json,
    metadata,
    min_observations,
    min_years,
    pl,
    require_valid_coords,
    temp_completeness,
):
    """Apply filtering criteria to metadata."""

    filtered = metadata

    # Temperature completeness filter
    if "temperature_completeness_pct" in filtered.columns:
        filtered = filtered.filter(
            pl.col("temperature_completeness_pct") >= temp_completeness.value
        )

    # Time span filter (calculate from years_available JSON)
    if "years_available" in filtered.columns:
        def count_years(years_json):
            if years_json is None:
                return 0
            try:
                years = json.loads(years_json) if isinstance(years_json, str) else years_json
                return len(years) if years else 0
            except Exception:
                return 0

        filtered = filtered.with_columns(
            pl.col("years_available").map_elements(
                count_years, return_dtype=pl.Int64
            ).alias("year_count")
        ).filter(pl.col("year_count") >= min_years.value)

    # Observation count filter
    if "total_observation_count" in filtered.columns:
        filtered = filtered.filter(
            pl.col("total_observation_count") >= min_observations.value
        )

    # Valid coordinates filter
    if require_valid_coords.value:
        filtered = filtered.filter(
            pl.col("latitude").is_not_null() &
            pl.col("longitude").is_not_null() &
            (pl.col("latitude") >= -90) & (pl.col("latitude") <= 90) &
            (pl.col("longitude") >= -180) & (pl.col("longitude") <= 180)
        )
        if "coord_valid" in filtered.columns:
            filtered = filtered.filter(pl.col("coord_valid"))

    n_filtered = len(filtered)
    n_original = len(metadata)
    n_removed = n_original - n_filtered
    pct_removed = 100 * n_removed / n_original if n_original > 0 else 0
    return filtered, n_filtered, n_original, n_removed, pct_removed


@app.cell
def show_filter_results(mo, n_filtered, n_original, n_removed, pct_removed):
    """Display filtering results."""
    mo.md(f"""
    ### Filtering Results

    - **Original stations**: {n_original:,}
    - **After filtering**: {n_filtered:,}
    - **Filtered out**: {n_removed:,} ({pct_removed:.1f}%)
    """)
    return


@app.cell
def create_histograms(filtered, mo, px):
    """Create and display histograms of key metrics."""
    fig_temp = None
    fig_obs = None

    # Temperature completeness histogram
    if "temperature_completeness_pct" in filtered.columns and len(filtered) > 0:
        fig_temp = px.histogram(
            filtered.to_pandas(),
            x="temperature_completeness_pct",
            nbins=30,
            title="Temperature Completeness Distribution",
            labels={"temperature_completeness_pct": "Completeness (%)"}
        )
        fig_temp.update_layout(showlegend=False, height=350)

    # Observation count histogram
    if "total_observation_count" in filtered.columns and len(filtered) > 0:
        fig_obs = px.histogram(
            filtered.to_pandas(),
            x="total_observation_count",
            nbins=30,
            title="Observation Count Distribution",
            labels={"total_observation_count": "Total Observations"}
        )
        fig_obs.update_layout(showlegend=False, height=350)

    # Build output - last expression displays
    header = mo.md("## 3. Distribution Visualizations")
    if fig_temp is not None and fig_obs is not None:
        content = mo.hstack([fig_temp, fig_obs])
    elif fig_temp is not None:
        content = fig_temp
    elif fig_obs is not None:
        content = fig_obs
    else:
        content = mo.md("*No data available for histograms*")

    mo.vstack([header, content])
    return


@app.cell
def create_and_display_map(MarkerCluster, filtered, folium, mo):
    """Create and display interactive geographic map of stations."""
    elements = [mo.md("## 4. Geographic Coverage")]

    if len(filtered) == 0:
        elements.append(mo.md("*No stations to display on map*"))
    else:
        # Create base map centered on data
        lat_center = filtered["latitude"].mean()
        lon_center = filtered["longitude"].mean()

        station_map = folium.Map(
            location=[lat_center, lon_center],
            zoom_start=4,
            tiles="CartoDB positron"
        )

        # Add marker cluster for performance
        marker_cluster = MarkerCluster().add_to(station_map)

        # Add markers (limit to 2000 for performance)
        sample = filtered.head(4458)
        for row in sample.iter_rows(named=True):
            lat, lon = row.get("latitude"), row.get("longitude")
            if lat is not None and lon is not None:
                station_id = row.get("station_id", "N/A")
                station_name = row.get("station_name", "N/A")
                temp_pct = row.get("temperature_completeness_pct", 0)
                obs_count = row.get("total_observation_count", 0)

                popup_text = f"""
                <b>Station:</b> {station_id}<br>
                <b>Name:</b> {station_name}<br>
                <b>Temp Completeness:</b> {temp_pct:.1f}%<br>
                <b>Observations:</b> {obs_count:,}
                """
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    popup=popup_text,
                    tooltip=station_id,
                    color="blue",
                    fill=True,
                    fill_opacity=0.6,
                ).add_to(marker_cluster)

        if len(filtered) > 2000:
            elements.append(mo.md(f"*Showing 2,000 of {len(filtered):,} stations for performance*"))
        elements.append(mo.Html(station_map._repr_html_()))

    mo.vstack(elements)
    return


@app.cell
def clustering_controls(mo):
    """Clustering configuration controls."""
    n_clusters = mo.ui.slider(
        start=2, stop=15, value=5, step=1,
        label="Number of Clusters (K-Means)"
    )

    cluster_features = mo.ui.multiselect(
        options=[
            "latitude",
            "longitude",
            "temperature_completeness_pct",
            "total_observation_count",
            "temperature_mean",
            "temperature_std",
        ],
        value=["latitude", "longitude", "temperature_completeness_pct"],
        label="Clustering Features"
    )
    return cluster_features, n_clusters


@app.cell
def perform_clustering(
    KMeans,
    StandardScaler,
    cluster_features,
    filtered,
    n_clusters,
    pl,
):
    """Perform K-Means clustering on filtered stations."""

    clustered_full = None
    kmeans_model = None
    cluster_error = None

    selected_features = cluster_features.value

    if len(selected_features) < 2:
        cluster_error = "Select at least 2 features for clustering"
    elif len(filtered) < n_clusters.value:
        cluster_error = f"Not enough stations ({len(filtered)}) for {n_clusters.value} clusters"
    else:
        # Check which features are available
        available_features = [f for f in selected_features if f in filtered.columns]

        if len(available_features) < 2:
            cluster_error = f"Not enough features available in data. Found: {available_features}"
        else:
            # Extract feature matrix (drop nulls)
            # Use set to avoid duplicate columns when lat/lon are in available_features
            select_cols = list(dict.fromkeys(
                ["station_id", "latitude", "longitude"] + available_features
            ))
            cluster_data = filtered.select(select_cols).drop_nulls()

            if len(cluster_data) < n_clusters.value:
                cluster_error = (
                    f"Not enough valid data points ({len(cluster_data)}) "
                    "after removing nulls"
                )
            else:
                X = cluster_data.select(available_features).to_numpy()

                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Fit K-Means
                kmeans_model = KMeans(
                    n_clusters=n_clusters.value,
                    random_state=42,
                    n_init=10
                )
                cluster_labels = kmeans_model.fit_predict(X_scaled)

                # Add cluster labels to data
                clustered = cluster_data.with_columns(
                    pl.Series("cluster", cluster_labels)
                )

                # Merge back with full filtered data
                clustered_full = filtered.join(
                    clustered.select(["station_id", "cluster"]),
                    on="station_id",
                    how="left"
                )

    n_clustered = len(clustered_full) if clustered_full is not None else 0
    features_used = available_features if clustered_full is not None else []
    return cluster_error, clustered_full, features_used, n_clustered


@app.cell
def show_clustering_status(
    cluster_error,
    features_used,
    mo,
    n_clustered,
    n_clusters,
):
    """Display clustering status."""
    mo.md(
        f"**Clustering Error**: {cluster_error}"
        if cluster_error
        else f"""
        ### Clustering Results

        - **Stations clustered**: {n_clustered:,}
        - **Features used**: {', '.join(features_used)}
        - **Clusters**: {n_clusters.value}
        """
    )
    return


@app.cell
def cluster_visualization(clustered_full, mo, pl, px):
    """Visualize clusters on map."""
    _output = None

    if clustered_full is None or "cluster" not in clustered_full.columns:
        _output = mo.md("*No clustering results to display*")
    else:
        # Scatter plot of clusters
        df_pd = clustered_full.filter(pl.col("cluster").is_not_null()).to_pandas()

        if len(df_pd) == 0:
            _output = mo.md("*No clustering results to display*")
        else:
            cluster_fig = px.scatter_geo(
                df_pd,
                lat="latitude",
                lon="longitude",
                color="cluster",
                hover_name="station_id",
                hover_data=["station_name", "temperature_completeness_pct"],
                title="Station Clusters (Geographic)",
                color_continuous_scale="Viridis",
            )
            cluster_fig.update_layout(height=500)
            cluster_fig.update_geos(
                showland=True,
                landcolor="lightgray",
                showocean=True,
                oceancolor="aliceblue",
                showcountries=True,
                countrycolor="white",
            )
            _output = cluster_fig

    return _output


@app.cell
def cluster_stats(clustered_full, pl):
    """Compute statistics per cluster."""

    if clustered_full is None or "cluster" not in clustered_full.columns:
        cluster_stats_df = None
    else:
        cluster_stats_df = (
            clustered_full
            .filter(pl.col("cluster").is_not_null())
            .group_by("cluster")
            .agg([
                pl.count().alias("station_count"),
                pl.col("temperature_completeness_pct").mean().alias("avg_temp_completeness"),
                pl.col("total_observation_count").mean().alias("avg_observations"),
                pl.col("latitude").mean().alias("centroid_lat"),
                pl.col("longitude").mean().alias("centroid_lon"),
            ])
            .sort("cluster")
        )
    return (cluster_stats_df,)


@app.cell
def display_cluster_stats(cluster_stats_df, mo, pl):
    """Display cluster statistics table."""
    if cluster_stats_df is not None and len(cluster_stats_df) > 0:
        # Format for display
        display_df = cluster_stats_df.with_columns([
            pl.col("avg_temp_completeness").round(1),
            pl.col("avg_observations").round(0).cast(pl.Int64),
            pl.col("centroid_lat").round(2),
            pl.col("centroid_lon").round(2),
        ])
        _output = mo.vstack([
            mo.md("### Cluster Statistics"),
            mo.ui.table(display_df.to_pandas()),
        ])
    else:
        _output = mo.md("*No cluster statistics available*")

    return _output


@app.cell
def export_controls(PROCESSED_DIR, filtered, mo):
    """Export filtered stations control."""
    export_path = PROCESSED_DIR / "filtered_stations.csv"

    export_button = mo.ui.run_button(label="Export to CSV")

    if export_button.value:
        filtered.write_csv(export_path)
        export_status = f"Exported {len(filtered):,} stations to `{export_path}`"
    else:
        export_status = f"Click to export {len(filtered):,} stations to `{export_path}`"

    mo.vstack([
        mo.md("## 6. Export Filtered Stations"),
        export_button,
        mo.md(export_status),
    ])
    return


if __name__ == "__main__":
    app.run()
