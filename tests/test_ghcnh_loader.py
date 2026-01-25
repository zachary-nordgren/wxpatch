"""Tests for GHCNh data loading and extraction utilities."""

import polars as pl
import pytest

from weather_imputation.data.ghcnh_loader import (
    PRIMARY_COLUMNS,
    TIER1_VARIABLES,
    VARIABLE_SUFFIXES,
    extract_tier1_variables,
)


@pytest.fixture
def sample_ghcnh_data() -> pl.DataFrame:
    """Create a sample GHCNh DataFrame for testing.

    Mimics the structure of GHCNh parquet files with primary columns
    and weather variables with their 6 attributes.
    """
    return pl.DataFrame({
        # Primary columns
        "STATION": ["USW00003046", "USW00003046", "USW00003046"],
        "Station_name": ["TEST STATION", "TEST STATION", "TEST STATION"],
        "DATE": ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 02:00"],
        "LATITUDE": [40.7128, 40.7128, 40.7128],
        "LONGITUDE": [-74.0060, -74.0060, -74.0060],
        # Temperature with all attributes
        "temperature": [15.5, 14.8, 14.2],
        "temperature_Quality_Code": [1, 1, 1],
        "temperature_Measurement_Code": [0, 0, 0],
        "temperature_Report_Type_Code": ["FM-15", "FM-15", "FM-15"],
        "temperature_Source_Code": [1, 1, 1],
        "temperature_units": ["Celsius", "Celsius", "Celsius"],
        # Dew point temperature with all attributes
        "dew_point_temperature": [10.2, 9.8, 9.5],
        "dew_point_temperature_Quality_Code": [1, 1, 1],
        "dew_point_temperature_Measurement_Code": [0, 0, 0],
        "dew_point_temperature_Report_Type_Code": ["FM-15", "FM-15", "FM-15"],
        "dew_point_temperature_Source_Code": [1, 1, 1],
        "dew_point_temperature_units": ["Celsius", "Celsius", "Celsius"],
        # Sea level pressure with all attributes
        "sea_level_pressure": [1013.25, 1013.50, 1013.75],
        "sea_level_pressure_Quality_Code": [1, 1, 1],
        "sea_level_pressure_Measurement_Code": [0, 0, 0],
        "sea_level_pressure_Report_Type_Code": ["FM-15", "FM-15", "FM-15"],
        "sea_level_pressure_Source_Code": [1, 1, 1],
        "sea_level_pressure_units": ["hPa", "hPa", "hPa"],
        # Wind speed with all attributes
        "wind_speed": [5.5, 6.2, 4.8],
        "wind_speed_Quality_Code": [1, 1, 1],
        "wind_speed_Measurement_Code": [0, 0, 0],
        "wind_speed_Report_Type_Code": ["FM-15", "FM-15", "FM-15"],
        "wind_speed_Source_Code": [1, 1, 1],
        "wind_speed_units": ["m/s", "m/s", "m/s"],
        # Wind direction with all attributes
        "wind_direction": [180.0, 185.0, 190.0],
        "wind_direction_Quality_Code": [1, 1, 1],
        "wind_direction_Measurement_Code": [0, 0, 0],
        "wind_direction_Report_Type_Code": ["FM-15", "FM-15", "FM-15"],
        "wind_direction_Source_Code": [1, 1, 1],
        "wind_direction_units": ["degrees", "degrees", "degrees"],
        # Relative humidity with all attributes
        "relative_humidity": [65.0, 68.0, 70.0],
        "relative_humidity_Quality_Code": [1, 1, 1],
        "relative_humidity_Measurement_Code": [0, 0, 0],
        "relative_humidity_Report_Type_Code": ["FM-15", "FM-15", "FM-15"],
        "relative_humidity_Source_Code": [1, 1, 1],
        "relative_humidity_units": ["%", "%", "%"],
        # Additional column not in Tier 1 (should be excluded)
        "visibility": [10000.0, 9500.0, 9000.0],
    })


def test_extract_tier1_variables_all_default(sample_ghcnh_data: pl.DataFrame) -> None:
    """Test extracting all Tier 1 variables with default parameters."""
    result = extract_tier1_variables(sample_ghcnh_data)

    # Should include primary columns
    for col in PRIMARY_COLUMNS:
        assert col in result.columns, f"Primary column {col} missing"

    # Should include all Tier 1 variables and their attributes
    for var in TIER1_VARIABLES:
        # Check that the base variable column exists
        assert var in result.columns, f"Variable {var} missing"

        # Check that all attribute columns exist
        for suffix in VARIABLE_SUFFIXES:
            if suffix:  # Skip empty suffix (already checked base variable)
                col_name = f"{var}{suffix}"
                assert col_name in result.columns, f"Attribute {col_name} missing"

    # Should exclude non-Tier1 variables (visibility is Tier 2)
    assert "visibility" not in result.columns

    # Should have same number of rows
    assert result.height == sample_ghcnh_data.height


def test_extract_tier1_variables_subset(sample_ghcnh_data: pl.DataFrame) -> None:
    """Test extracting a subset of Tier 1 variables."""
    variables = ["temperature", "wind_speed"]
    result = extract_tier1_variables(sample_ghcnh_data, variables=variables)

    # Should include primary columns
    for col in PRIMARY_COLUMNS:
        assert col in result.columns

    # Should include only requested variables
    assert "temperature" in result.columns
    assert "temperature_Quality_Code" in result.columns
    assert "wind_speed" in result.columns
    assert "wind_speed_Quality_Code" in result.columns

    # Should not include unrequested variables
    assert "dew_point_temperature" not in result.columns
    assert "sea_level_pressure" not in result.columns

    # Should have same number of rows
    assert result.height == sample_ghcnh_data.height


def test_extract_tier1_variables_single_variable(sample_ghcnh_data: pl.DataFrame) -> None:
    """Test extracting a single variable."""
    result = extract_tier1_variables(sample_ghcnh_data, variables=["temperature"])

    # Should include primary columns + temperature + 5 attributes = 11 columns
    expected_cols = len(PRIMARY_COLUMNS) + len(VARIABLE_SUFFIXES)
    assert len(result.columns) == expected_cols

    # Verify temperature columns
    assert "temperature" in result.columns
    assert "temperature_Quality_Code" in result.columns
    assert "temperature_Measurement_Code" in result.columns
    assert "temperature_Report_Type_Code" in result.columns
    assert "temperature_Source_Code" in result.columns
    assert "temperature_units" in result.columns


def test_extract_tier1_variables_missing_columns() -> None:
    """Test extraction when some variable columns are missing."""
    # Create DataFrame with only some attributes
    df = pl.DataFrame({
        "STATION": ["USW00003046"],
        "Station_name": ["TEST STATION"],
        "DATE": ["2023-01-01 00:00"],
        "LATITUDE": [40.7128],
        "LONGITUDE": [-74.0060],
        "temperature": [15.5],
        "temperature_Quality_Code": [1],
        # Missing other temperature attributes
    })

    result = extract_tier1_variables(df, variables=["temperature"])

    # Should include only columns that exist
    assert "temperature" in result.columns
    assert "temperature_Quality_Code" in result.columns
    # Should not fail on missing columns
    assert "temperature_Measurement_Code" not in result.columns


def test_extract_tier1_variables_empty_dataframe() -> None:
    """Test extraction from empty DataFrame."""
    empty_df = pl.DataFrame()
    result = extract_tier1_variables(empty_df)

    # Should return empty DataFrame
    assert result.is_empty()


def test_extract_tier1_variables_no_matching_columns() -> None:
    """Test extraction when no requested columns exist."""
    df = pl.DataFrame({
        "random_column": [1, 2, 3],
        "another_column": ["a", "b", "c"],
    })

    result = extract_tier1_variables(df)

    # Should return empty DataFrame
    assert result.is_empty()


def test_extract_tier1_variables_preserves_data_values(sample_ghcnh_data: pl.DataFrame) -> None:
    """Test that extraction preserves original data values."""
    result = extract_tier1_variables(sample_ghcnh_data, variables=["temperature"])

    # Check that values match original
    assert result["temperature"].to_list() == [15.5, 14.8, 14.2]
    assert result["temperature_Quality_Code"].to_list() == [1, 1, 1]
    assert result["STATION"].to_list() == ["USW00003046", "USW00003046", "USW00003046"]


def test_extract_tier1_variables_preserves_row_order(sample_ghcnh_data: pl.DataFrame) -> None:
    """Test that extraction preserves row order."""
    result = extract_tier1_variables(sample_ghcnh_data)

    # DATE should be in same order
    assert result["DATE"].to_list() == [
        "2023-01-01 00:00",
        "2023-01-01 01:00",
        "2023-01-01 02:00",
    ]


def test_extract_tier1_variables_with_nulls() -> None:
    """Test extraction handles null values correctly."""
    df = pl.DataFrame({
        "STATION": ["USW00003046", "USW00003046"],
        "Station_name": ["TEST STATION", "TEST STATION"],
        "DATE": ["2023-01-01 00:00", "2023-01-01 01:00"],
        "LATITUDE": [40.7128, 40.7128],
        "LONGITUDE": [-74.0060, -74.0060],
        "temperature": [15.5, None],
        "temperature_Quality_Code": [1, None],
        "temperature_Measurement_Code": [0, None],
        "temperature_Report_Type_Code": ["FM-15", None],
        "temperature_Source_Code": [1, None],
        "temperature_units": ["Celsius", None],
    })

    result = extract_tier1_variables(df, variables=["temperature"])

    # Should preserve nulls
    assert result["temperature"][1] is None
    assert result["temperature_Quality_Code"][1] is None
    assert result.height == 2


def test_tier1_variables_constant_matches_spec() -> None:
    """Test that TIER1_VARIABLES matches SPEC.md requirements."""
    expected_tier1 = [
        "temperature",
        "dew_point_temperature",
        "sea_level_pressure",
        "wind_direction",
        "wind_speed",
        "relative_humidity",
    ]

    # Should match exactly (order matters for consistency)
    assert expected_tier1 == TIER1_VARIABLES


def test_variable_suffixes_constant_complete() -> None:
    """Test that VARIABLE_SUFFIXES contains all expected attributes."""
    expected_suffixes = [
        "",  # Base value
        "_Quality_Code",
        "_Measurement_Code",
        "_Report_Type_Code",
        "_Source_Code",
        "_units",
    ]

    assert expected_suffixes == VARIABLE_SUFFIXES
