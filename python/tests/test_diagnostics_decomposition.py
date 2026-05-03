"""Tests for factor decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import DecompositionReport


def test_factor_decomposition_market_only():
    """Strategy returns = 0.5 * market → beta ≈ 0.5, alpha ≈ 0."""
    from skuld_research.diagnostics.decomposition import factor_decomposition

    np.random.seed(42)
    n_months = 60

    # Market returns
    market = pd.Series(
        np.random.randn(n_months) * 0.03,
        index=pd.date_range("2020-01-31", periods=n_months, freq="BME"),
    )

    # Strategy = 0.5 * market + small noise
    strategy = 0.5 * market + pd.Series(
        np.random.randn(n_months) * 0.005,
        index=market.index,
    )

    result = factor_decomposition(
        strategy_returns=strategy,
        market_returns=market,
        factor_returns_dict={},
    )

    assert isinstance(result, DecompositionReport)
    assert "market" in result.regressors
    assert "market" in result.coefficients
    # Beta on market should be close to 0.5
    assert 0.4 < result.coefficients["market"] < 0.6
    # Alpha should be near zero
    assert abs(result.residual_alpha_annualised) < 0.05  # <5% annualized
    assert result.n_obs == n_months


def test_factor_decomposition_factor_loading():
    """Strategy returns = 1.0 * factor → beta ≈ 1.0 on that factor."""
    from skuld_research.diagnostics.decomposition import factor_decomposition

    np.random.seed(99)
    n_months = 80

    dates = pd.date_range("2018-01-31", periods=n_months, freq="BME")

    # Market returns (uncorrelated with strategy)
    market = pd.Series(np.random.randn(n_months) * 0.02, index=dates)

    # Factor returns
    momentum_factor = pd.Series(np.random.randn(n_months) * 0.04, index=dates)

    # Strategy = 1.0 * momentum_factor + small noise
    strategy = momentum_factor + pd.Series(np.random.randn(n_months) * 0.005, index=dates)

    result = factor_decomposition(
        strategy_returns=strategy,
        market_returns=market,
        factor_returns_dict={"momentum": momentum_factor},
    )

    assert "market" in result.regressors
    assert "momentum" in result.regressors
    # Beta on momentum should be close to 1.0
    assert 0.9 < result.coefficients["momentum"] < 1.1
    # Beta on market should be close to 0
    assert abs(result.coefficients["market"]) < 0.2
    # Alpha should be near zero
    assert abs(result.residual_alpha_annualised) < 0.05


def test_factor_decomposition_r_squared():
    """R^2 should be high when strategy is fully explained by factors."""
    from skuld_research.diagnostics.decomposition import factor_decomposition

    np.random.seed(123)
    n_months = 100

    dates = pd.date_range("2015-01-31", periods=n_months, freq="BME")

    market = pd.Series(np.random.randn(n_months) * 0.03, index=dates)
    momentum = pd.Series(np.random.randn(n_months) * 0.04, index=dates)

    # Strategy = 0.6 * market + 0.4 * momentum (no noise)
    strategy = 0.6 * market + 0.4 * momentum

    result = factor_decomposition(
        strategy_returns=strategy,
        market_returns=market,
        factor_returns_dict={"momentum": momentum},
    )

    # R^2 should be very high (near 1.0) since strategy is perfectly explained
    assert result.r_squared > 0.95
    assert result.coefficients["market"] > 0.5
    assert result.coefficients["momentum"] > 0.3


def test_factor_decomposition_newey_west_t_stats():
    """T-stats should be finite and reasonable."""
    from skuld_research.diagnostics.decomposition import factor_decomposition

    np.random.seed(456)
    n_months = 50

    dates = pd.date_range("2019-01-31", periods=n_months, freq="BME")

    market = pd.Series(np.random.randn(n_months) * 0.025, index=dates)
    value = pd.Series(np.random.randn(n_months) * 0.03, index=dates)

    # Strategy with clear exposure to both
    strategy = 0.7 * market + 0.5 * value + pd.Series(np.random.randn(n_months) * 0.01, index=dates)

    result = factor_decomposition(
        strategy_returns=strategy,
        market_returns=market,
        factor_returns_dict={"value": value},
    )

    # T-stats should be finite
    assert not np.isnan(result.t_stats["market"])
    assert not np.isnan(result.t_stats["value"])
    assert not np.isnan(result.residual_alpha_t_stat)
    # With 50 observations and clear exposures, t-stats should be significant
    assert abs(result.t_stats["market"]) > 2.0
    assert abs(result.t_stats["value"]) > 2.0


def test_factor_decomposition_handles_missing_alignment():
    """Decomposition handles misaligned indices correctly."""
    from skuld_research.diagnostics.decomposition import factor_decomposition

    # Strategy has 10 months
    strategy = pd.Series(
        np.random.randn(10) * 0.02,
        index=pd.date_range("2020-01-31", periods=10, freq="BME"),
    )

    # Market has 12 months (extends beyond strategy)
    market = pd.Series(
        np.random.randn(12) * 0.02,
        index=pd.date_range("2020-01-31", periods=12, freq="BME"),
    )

    # Factor has 8 months (shorter than strategy)
    factor = pd.Series(
        np.random.randn(8) * 0.03,
        index=pd.date_range("2020-01-31", periods=8, freq="BME"),
    )

    result = factor_decomposition(
        strategy_returns=strategy,
        market_returns=market,
        factor_returns_dict={"test_factor": factor},
    )

    # Should use intersection (8 months)
    assert result.n_obs == 8
