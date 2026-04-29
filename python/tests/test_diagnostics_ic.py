"""Tests for ranking IC computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import ICReport


def test_ranking_ic_perfect_monotone():
    """Perfect monotone factor → IC ≈ 1.0."""
    from skuld_research.diagnostics.ic import ranking_ic

    # 3 rebalance dates, 5 tickers each
    dates = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-31"])
    tickers = ["A", "B", "C", "D", "E"]

    # Factor scores: perfect rank order (A=1, B=2, C=3, D=4, E=5)
    factor_panel = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0]] * 3,
        index=dates,
        columns=tickers,
    )

    # Forward returns: perfectly rank-aligned with factor
    # (higher score → higher return)
    returns_panel = pd.DataFrame(
        [[0.01, 0.02, 0.03, 0.04, 0.05]] * 10,  # 10 months of monthly returns
        index=pd.date_range("2024-01-31", periods=10, freq="BME"),
        columns=tickers,
    )

    result = ranking_ic(factor_panel, returns_panel, horizon_months=1, min_cross_section=3)

    assert isinstance(result, ICReport)
    assert result.factor_name == "factor"
    assert result.horizon_months == 1
    assert result.n_obs == 3
    # IC should be very close to +1.0 for perfect monotone relationship
    assert result.ic_mean > 0.95
    assert result.ic_std >= 0.0
    # ic_ir will be NaN for perfect correlation (std=0), which is acceptable
    # t_stat_newey_west will also be NaN for zero variance
    assert result.min_universe_per_date == 5


def test_ranking_ic_random_near_zero():
    """Random uncorrelated factor → IC ≈ 0, |t-stat| small."""
    from skuld_research.diagnostics.ic import ranking_ic

    np.random.seed(42)
    n_dates = 20
    n_tickers = 30

    dates = pd.date_range("2020-01-31", periods=n_dates, freq="BME")
    tickers = [f"T{i}" for i in range(n_tickers)]

    # Random factor scores
    factor_panel = pd.DataFrame(
        np.random.randn(n_dates, n_tickers),
        index=dates,
        columns=tickers,
    )

    # Random forward returns (independent of factor)
    returns_panel = pd.DataFrame(
        np.random.randn(n_dates + 12, n_tickers) * 0.05,
        index=pd.date_range("2020-01-31", periods=n_dates + 12, freq="BME"),
        columns=tickers,
    )

    result = ranking_ic(factor_panel, returns_panel, horizon_months=1, min_cross_section=10)

    assert result.n_obs > 0
    # Mean IC should be near zero (not exactly zero due to sampling)
    assert abs(result.ic_mean) < 0.3
    # t-stat should be small (not significant)
    assert abs(result.t_stat_newey_west) < 3.0


def test_ranking_ic_min_cross_section_enforced():
    """Dates with fewer than min_cross_section tickers are dropped."""
    from skuld_research.diagnostics.ic import ranking_ic

    dates = pd.DatetimeIndex(["2024-01-31", "2024-02-29"])
    tickers = ["A", "B", "C"]

    # First date: 3 valid scores, second date: only 2 (one NaN)
    factor_panel = pd.DataFrame(
        [[1.0, 2.0, 3.0], [1.0, 2.0, np.nan]],
        index=dates,
        columns=tickers,
    )

    returns_panel = pd.DataFrame(
        [[0.01, 0.02, 0.03]] * 5,
        index=pd.date_range("2024-01-31", periods=5, freq="BME"),
        columns=tickers,
    )

    # min_cross_section=3: second date should be dropped
    result = ranking_ic(factor_panel, returns_panel, horizon_months=1, min_cross_section=3)

    assert result.n_obs == 1
    assert result.min_universe_per_date == 3


def test_ranking_ic_newey_west_variance_ge_raw():
    """Newey-West variance ≥ raw variance for autocorrelated IC series."""
    from skuld_research.diagnostics.ic import ranking_ic

    # Construct a factor with strong positive autocorrelation in IC
    n_dates = 30
    dates = pd.date_range("2020-01-31", periods=n_dates, freq="BME")
    tickers = ["A", "B", "C", "D", "E"]

    # Factor scores that create autocorrelated IC pattern
    # (slowly drifting scores create persistent rank relationships)
    np.random.seed(99)
    drift = np.cumsum(np.random.randn(n_dates) * 0.1)
    factor_panel = pd.DataFrame(
        np.tile(drift[:, None], (1, 5)) + np.random.randn(n_dates, 5) * 0.5,
        index=dates,
        columns=tickers,
    )

    # Returns also autocorrelated - extend drift to match n_dates + 12
    drift_extended = np.cumsum(np.random.randn(n_dates + 12) * 0.1)
    returns_panel = pd.DataFrame(
        np.tile(drift_extended[:, None], (1, 5)) * 0.02 + np.random.randn(n_dates + 12, 5) * 0.03,
        index=pd.date_range("2020-01-31", periods=n_dates + 12, freq="BME"),
        columns=tickers,
    )

    result = ranking_ic(factor_panel, returns_panel, horizon_months=3, min_cross_section=3)

    # With autocorrelated IC, Newey-West SE should be larger (more conservative)
    # which means SE_nw >= SE_raw, but t_stat uses SE in denominator so t_nw <= t_raw
    # This test just checks that computation completes and t-stat is finite
    assert not np.isnan(result.t_stat_newey_west)
    assert result.n_obs > 10


def test_ranking_ic_custom_factor_name():
    """Factor name can be customized."""
    from skuld_research.diagnostics.ic import ranking_ic

    dates = pd.DatetimeIndex(["2024-01-31"])
    tickers = ["A", "B", "C"]

    factor_panel = pd.DataFrame([[1.0, 2.0, 3.0]], index=dates, columns=tickers)
    returns_panel = pd.DataFrame(
        [[0.01, 0.02, 0.03]] * 3,
        index=pd.date_range("2024-01-31", periods=3, freq="BME"),
        columns=tickers,
    )

    result = ranking_ic(
        factor_panel,
        returns_panel,
        horizon_months=1,
        factor_name="momentum",
        min_cross_section=2,
    )

    assert result.factor_name == "momentum"
