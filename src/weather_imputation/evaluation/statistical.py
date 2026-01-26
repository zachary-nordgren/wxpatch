"""Statistical significance tests for comparing imputation methods."""

import math
from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass
class PairedTestResult:
    """Result of a paired statistical test."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool  # At alpha=0.05
    effect_size: float | None
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "effect_size": self.effect_size,
            "interpretation": self.interpretation,
        }


def paired_t_test(
    errors_a: pl.Series,
    errors_b: pl.Series,
    alpha: float = 0.05,
) -> PairedTestResult:
    """Perform paired t-test on two sets of errors.

    Tests whether the mean difference between paired observations is
    significantly different from zero.

    Args:
        errors_a: Errors from method A
        errors_b: Errors from method B
        alpha: Significance level

    Returns:
        PairedTestResult with test statistics
    """
    # Filter to pairs where both are present
    mask = errors_a.is_not_null() & errors_b.is_not_null()
    a = errors_a.filter(mask)
    b = errors_b.filter(mask)

    n = len(a)
    if n < 2:
        return PairedTestResult(
            test_name="paired_t_test",
            statistic=float("nan"),
            p_value=1.0,
            significant=False,
            effect_size=None,
            interpretation="Insufficient data for t-test",
        )

    # Calculate differences
    diff = a - b
    mean_diff = diff.mean()
    std_diff = diff.std()

    if std_diff == 0:
        return PairedTestResult(
            test_name="paired_t_test",
            statistic=float("nan"),
            p_value=1.0,
            significant=False,
            effect_size=None,
            interpretation="No variance in differences",
        )

    # t-statistic
    t_stat = mean_diff / (std_diff / math.sqrt(n))

    # TODO: Calculate p-value using t-distribution
    # For now, use approximation for large n (degrees of freedom = n - 1)
    p_value = 2 * (1 - _normal_cdf(abs(t_stat))) if n > 30 else float("nan")

    # Cohen's d effect size
    effect_size = mean_diff / std_diff if std_diff > 0 else None

    significant = p_value < alpha if not math.isnan(p_value) else False

    if significant:
        better = "A" if mean_diff < 0 else "B"
        interpretation = f"Method {better} is significantly better (p={p_value:.4f})"
    else:
        interpretation = f"No significant difference (p={p_value:.4f})"

    return PairedTestResult(
        test_name="paired_t_test",
        statistic=t_stat,
        p_value=p_value,
        significant=significant,
        effect_size=effect_size,
        interpretation=interpretation,
    )


def wilcoxon_signed_rank_test(
    errors_a: pl.Series,
    errors_b: pl.Series,
    alpha: float = 0.05,
) -> PairedTestResult:
    """Perform Wilcoxon signed-rank test.

    Non-parametric alternative to paired t-test. Does not assume
    normal distribution of differences.

    Args:
        errors_a: Errors from method A
        errors_b: Errors from method B
        alpha: Significance level

    Returns:
        PairedTestResult with test statistics
    """
    # TODO: Implement Wilcoxon signed-rank test
    # This requires ranking and computing the test statistic
    return PairedTestResult(
        test_name="wilcoxon_signed_rank",
        statistic=float("nan"),
        p_value=1.0,
        significant=False,
        effect_size=None,
        interpretation="Not yet implemented",
    )


def diebold_mariano_test(
    errors_a: pl.Series,
    errors_b: pl.Series,
    alpha: float = 0.05,
    loss_func: str = "squared",
) -> PairedTestResult:
    """Perform Diebold-Mariano test for forecast comparison.

    Tests whether two forecasts have equal predictive accuracy.

    Args:
        errors_a: Errors from method A
        errors_b: Errors from method B
        alpha: Significance level
        loss_func: Loss function - "squared" or "absolute"

    Returns:
        PairedTestResult with test statistics
    """
    mask = errors_a.is_not_null() & errors_b.is_not_null()
    a = errors_a.filter(mask)
    b = errors_b.filter(mask)

    n = len(a)
    if n < 2:
        return PairedTestResult(
            test_name="diebold_mariano",
            statistic=float("nan"),
            p_value=1.0,
            significant=False,
            effect_size=None,
            interpretation="Insufficient data",
        )

    # Calculate loss differential
    d = (a ** 2) - (b ** 2) if loss_func == "squared" else a.abs() - b.abs()

    mean_d = d.mean()
    var_d = d.var()

    if var_d == 0:
        return PairedTestResult(
            test_name="diebold_mariano",
            statistic=float("nan"),
            p_value=1.0,
            significant=False,
            effect_size=None,
            interpretation="No variance in loss differential",
        )

    # DM statistic (simplified, assuming no autocorrelation)
    dm_stat = mean_d / math.sqrt(var_d / n)

    # Approximate p-value using normal distribution
    p_value = 2 * (1 - _normal_cdf(abs(dm_stat)))

    significant = p_value < alpha

    if significant:
        better = "A" if mean_d < 0 else "B"
        interpretation = f"Method {better} has significantly lower {loss_func} error"
    else:
        interpretation = "No significant difference in predictive accuracy"

    return PairedTestResult(
        test_name="diebold_mariano",
        statistic=dm_stat,
        p_value=p_value,
        significant=significant,
        effect_size=None,
        interpretation=interpretation,
    )


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
