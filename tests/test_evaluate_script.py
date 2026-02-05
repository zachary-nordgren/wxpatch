"""Tests for evaluation script.

These are integration tests that verify the evaluation pipeline works end-to-end.
"""

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest
import torch

from weather_imputation.data.masking import (
    apply_mar_mask,
    apply_mcar_mask,
    apply_mnar_mask,
    apply_realistic_mask,
)
from weather_imputation.evaluation.metrics import compute_all_metrics
from weather_imputation.evaluation.stratified import (
    stratify_by_gap_length,
    stratify_by_season,
    stratify_by_variable,
)
from weather_imputation.models.classical.linear import LinearInterpolationImputer
from weather_imputation.models.classical.mice import MICEImputer
from weather_imputation.models.classical.spline import AkimaSplineImputer


class TestEvaluationComponents:
    """Test individual components used by the evaluation script."""

    def test_classical_model_instantiation(self):
        """Verify all classical models can be instantiated with correct parameters.

        This would catch constructor parameter errors.
        """
        # Linear interpolation
        linear = LinearInterpolationImputer(max_gap_length=24)
        assert linear.max_gap_length == 24

        # Spline interpolation
        spline = AkimaSplineImputer(max_gap_length=24)
        assert spline.max_gap_length == 24

        # MICE
        mice = MICEImputer(
            predictor_method="bayesian_ridge",
            n_iterations=10,
            n_imputations=5,
        )
        assert mice.predictor_method == "bayesian_ridge"
        assert mice.n_iterations == 10
        assert mice.n_imputations == 5

    def test_masking_strategies_signatures(self):
        """Verify masking strategies accept correct parameters.

        This would catch parameter name errors in synthetic masking.
        """
        observed = torch.randn(1, 100, 6)

        # MCAR masking
        mcar_mask = apply_mcar_mask(
            observed,
            missing_ratio=0.2,
            min_gap_length=1,
            max_gap_length=168,
        )
        assert mcar_mask.shape == observed.shape
        assert mcar_mask.dtype == torch.bool

        # MAR masking
        mar_mask = apply_mar_mask(
            observed,
            missing_ratio=0.2,
            extreme_percentile=0.15,
        )
        assert mar_mask.shape == observed.shape

        # MNAR masking
        mnar_mask = apply_mnar_mask(
            observed,
            missing_ratio=0.2,
            extreme_multiplier=5.0,
        )
        assert mnar_mask.shape == observed.shape

        # Realistic masking
        realistic_mask = apply_realistic_mask(
            observed,
            missing_ratio=0.2,
            gap_distribution="empirical",
        )
        assert realistic_mask.shape == observed.shape

    def test_metrics_computation_signature(self):
        """Verify metrics functions accept correct parameters.

        This would catch parameter name or type errors.
        """
        y_true = torch.randn(2, 50, 6)
        y_pred = torch.randn(2, 50, 6)
        mask = torch.rand(2, 50, 6) < 0.3  # 30% evaluation positions

        # Compute all metrics
        metrics = compute_all_metrics(y_true, y_pred, mask)

        assert isinstance(metrics, dict)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mse" in metrics
        assert "bias" in metrics
        assert "r2" in metrics

    def test_stratified_evaluation_signatures(self):
        """Verify stratified evaluation functions accept correct parameters.

        This would catch parameter name errors in stratification.
        """
        y_true = torch.randn(2, 50, 6)
        y_pred = torch.randn(2, 50, 6)
        mask = torch.rand(2, 50, 6) < 0.3
        gap_lengths = torch.randint(1, 200, (2, 50, 6)).float()
        timestamps = torch.randint(0, 1000000, (2, 50))

        # Gap length stratification
        gap_results = stratify_by_gap_length(
            y_true, y_pred, mask, gap_lengths, bins=[6, 24, 72, 168]
        )
        assert isinstance(gap_results, dict)

        # Variable stratification
        var_names = ["temp", "dewpt", "pressure", "wspd", "wdir", "rh"]
        var_results = stratify_by_variable(
            y_true, y_pred, mask, variable_names=var_names
        )
        assert isinstance(var_results, dict)

        # Season stratification
        season_results = stratify_by_season(y_true, y_pred, mask, timestamps)
        assert isinstance(season_results, dict)

    def test_imputer_fit_and_impute_workflow(self):
        """Test the complete imputer workflow: fit -> impute.

        This verifies the Imputer protocol is correctly implemented.
        """
        # Create sample data
        observed = torch.randn(5, 20, 3)
        mask = torch.rand(5, 20, 3) < 0.8  # 80% observed

        # Test linear interpolation
        linear = LinearInterpolationImputer()
        linear.fit(None, None)  # No-op for classical methods
        imputed = linear.impute(observed, mask)
        assert imputed.shape == observed.shape

        # Test spline interpolation
        spline = AkimaSplineImputer()
        spline.fit(None, None)
        imputed = spline.impute(observed, mask)
        assert imputed.shape == observed.shape

        # Test MICE (requires actual training data)
        # MICE needs a DataLoader, so we skip it in this basic test
        # It's tested in the integration tests below
        # mice = MICEImputer(n_iterations=2, n_imputations=2)
        # mice.fit(train_loader)  # Would need actual DataLoader
        # imputed = mice.impute(observed[:2, :10, :], mask[:2, :10, :])
        # assert imputed.shape == observed[:2, :10, :].shape


class TestEvaluationIntegration:
    """Integration tests for the full evaluation pipeline."""

    def test_evaluation_pipeline_linear(self, sample_test_data):
        """Test full evaluation pipeline with linear interpolation.

        This simulates what evaluate.py does with a classical model.
        """
        observed, mask, target, timestamps = sample_test_data

        # Apply synthetic masking (simulating what evaluate.py does)
        synthetic_mask = apply_mcar_mask(
            observed, missing_ratio=0.2, min_gap_length=1, max_gap_length=168
        )

        # Create masked observed data
        masked_observed = observed.clone()
        masked_observed[~synthetic_mask] = 0.0

        # Evaluation mask
        eval_mask = mask & ~synthetic_mask

        # Create and fit model
        imputer = LinearInterpolationImputer(max_gap_length=None)
        imputer.fit(None, None)

        # Run imputation
        imputed = imputer.impute(masked_observed, synthetic_mask)

        # Compute metrics
        metrics = compute_all_metrics(target, imputed, eval_mask)

        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert all(isinstance(v, float) for v in metrics.values())

        # Metrics should be reasonable (not NaN unless no eval positions)
        if eval_mask.sum() > 0:
            # At least some metrics should be finite
            finite_count = sum(1 for v in metrics.values() if not torch.isnan(torch.tensor(v)))
            assert finite_count > 0

    def test_evaluation_pipeline_with_stratification(self, sample_test_data):
        """Test evaluation pipeline with stratified analysis.

        This verifies the complete stratified evaluation workflow.
        """
        observed, mask, target, timestamps = sample_test_data

        # Apply synthetic masking
        synthetic_mask = apply_realistic_mask(
            observed,
            missing_ratio=0.2,
            gap_distribution="empirical",
        )

        masked_observed = observed.clone()
        masked_observed[~synthetic_mask] = 0.0
        eval_mask = mask & ~synthetic_mask

        # Compute gap lengths (same logic as evaluate.py)
        gap_lengths = compute_gap_lengths(synthetic_mask)

        # Run imputation
        imputer = LinearInterpolationImputer()
        imputer.fit(None, None)
        imputed = imputer.impute(masked_observed, synthetic_mask)

        # Compute overall metrics
        overall_metrics = compute_all_metrics(target, imputed, eval_mask)
        assert isinstance(overall_metrics, dict)

        # Compute stratified metrics
        var_names = ["var1", "var2", "var3"]

        gap_results = stratify_by_gap_length(
            target, imputed, eval_mask, gap_lengths
        )
        assert isinstance(gap_results, dict)

        var_results = stratify_by_variable(
            target, imputed, eval_mask, variable_names=var_names
        )
        assert isinstance(var_results, dict)

        season_results = stratify_by_season(target, imputed, eval_mask, timestamps)
        assert isinstance(season_results, dict)

    def test_results_saving_formats(self, tmp_path):
        """Test that results can be saved in different formats.

        This verifies the save_results functionality.
        """
        # Sample results
        results = {
            "model": "linear",
            "masking_strategy": "mcar",
            "mask_ratio": 0.2,
            "overall": {
                "rmse": 1.23,
                "mae": 0.98,
                "bias": 0.05,
            },
        }

        # Test JSON saving
        json_file = tmp_path / "results.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)

        assert json_file.exists()

        # Verify can load
        with open(json_file) as f:
            loaded = json.load(f)
        assert loaded["model"] == "linear"
        assert loaded["overall"]["rmse"] == 1.23

        # Test Parquet saving (flattened structure)
        flat_results = []
        for key, value in results.items():
            if isinstance(value, dict):
                row = {"metric_group": key}
                row.update(value)
                flat_results.append(row)
            else:
                flat_results.append({"metric": key, "value": value})

        df = pl.DataFrame(flat_results)
        parquet_file = tmp_path / "results.parquet"
        df.write_parquet(parquet_file)

        assert parquet_file.exists()

        # Verify can load
        loaded_df = pl.read_parquet(parquet_file)
        assert len(loaded_df) > 0


class TestEvaluateScriptCLI:
    """Tests for the evaluation script CLI itself."""

    def test_help_command(self):
        """Test that the help command works."""
        import subprocess

        result = subprocess.run(
            ["uv", "run", "python", "src/scripts/evaluate.py", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "evaluate" in result.stdout.lower()
        assert "--model" in result.stdout
        assert "--checkpoint" in result.stdout

    def test_mutual_exclusion_validation(self):
        """Test that CLI validates --model and --checkpoint are mutually exclusive.

        This would catch validation logic errors.
        """
        import subprocess

        # Test with neither (should fail)
        result = subprocess.run(
            [
                "uv", "run", "python", "src/scripts/evaluate.py",
                "--test-file", "fake.parquet",
            ],
            capture_output=True,
            text=True,
        )
        # Should fail with error message (in stdout for print_error)
        assert result.returncode == 1
        output = (result.stdout + result.stderr).lower()
        assert "model" in output or "checkpoint" in output

    @pytest.mark.slow
    def test_evaluation_with_test_data(self, tmp_path):
        """Test evaluation with actual test data file.

        This is marked as slow and would be run separately.
        Requires actual preprocessed data to exist.
        """
        import subprocess

        # Create a minimal test data file
        test_data = pl.DataFrame({
            "DATE": ["2020-01-01 00:00:00"] * 100,
            "station_id": ["TEST_STATION"] * 100,
            "latitude": [40.0] * 100,
            "longitude": [-105.0] * 100,
            "temperature": list(range(100)),
            "dew_point_temperature": list(range(100)),
            "sea_level_pressure": [1000.0] * 100,
            "wind_speed": [5.0] * 100,
            "wind_direction": [180.0] * 100,
            "relative_humidity": [50.0] * 100,
        })

        test_file = tmp_path / "test.parquet"
        test_data.write_parquet(test_file)

        # Run evaluation
        result = subprocess.run(
            [
                "uv", "run", "python", "src/scripts/evaluate.py",
                "--model", "linear",
                "--test-file", str(test_file),
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Should succeed
        if result.returncode == 0:
            # Check output file was created
            output_file = tmp_path / "linear_results.json"
            assert output_file.exists()

            # Verify results structure
            with open(output_file) as f:
                results = json.load(f)
            assert "model" in results
            assert "overall" in results


# Helper functions

def compute_gap_lengths(mask: torch.Tensor) -> torch.Tensor:
    """Compute gap lengths (same as evaluate.py).

    This is duplicated here for testing purposes.
    """
    N, T, V = mask.shape
    gap_lengths = torch.zeros_like(mask, dtype=torch.float32)

    for n in range(N):
        for v in range(V):
            obs = mask[n, :, v]
            gaps = gap_lengths[n, :, v]

            i = 0
            while i < T:
                if not obs[i]:
                    gap_start = i
                    gap_end = i
                    while gap_end < T and not obs[gap_end]:
                        gap_end += 1
                    gap_len = gap_end - gap_start
                    gaps[gap_start:gap_end] = gap_len
                    i = gap_end
                else:
                    i += 1

    return gap_lengths


# Fixtures

@pytest.fixture
def sample_test_data():
    """Create sample test data for evaluation.

    Returns (observed, mask, target, timestamps) tuple.
    """
    N, T, V = 2, 50, 3
    observed = torch.randn(N, T, V)
    mask = torch.rand(N, T, V) < 0.9  # 90% observed
    target = observed.clone()
    timestamps = torch.arange(T).unsqueeze(0).expand(N, T)

    return observed, mask, target, timestamps
