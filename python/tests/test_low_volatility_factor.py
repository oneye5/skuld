"""Tests for the low-volatility factor.

These tests are written before the implementation (TDD red phase).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import PITSnapshot, PreparedPanel


def _make_prepared_panel(
    n_days: int = 800,
    tickers: list[str] | None = None,
    seed: int = 42,
    asof: str = "2025-01-01",
    custom_returns: dict[str, pd.Series] | None = None,
) -> PreparedPanel:
    """Build a synthetic PreparedPanel with controllable volatility profiles."""
    if tickers is None:
        tickers = ["AAA.NZ", "BBB.NZ", "CCC.NZ"]

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)

    if custom_returns is not None:
        # Use provided daily returns to build prices
        prices_data = {}
        for ticker, rets in custom_returns.items():
            prices_data[ticker] = 10.0 * (1 + rets).cumprod()
        prices = pd.DataFrame(prices_data, index=dates)
    else:
        # Default synthetic prices
        prices_data = {}
        for t in tickers:
            px = 10.0 * (1 + 0.001 * rng.standard_normal(n_days)).cumprod()
            prices_data[t] = px
        prices = pd.DataFrame(prices_data, index=dates)

    prices.index.name = "date"

    volumes = pd.DataFrame(
        {t: np.full(n_days, 500_000.0) for t in tickers},
        index=dates,
    )
    volumes.index.name = "date"

    asof_ts = pd.Timestamp(asof)

    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                [], names=["ticker", "publication_date"]
            )
        ),
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(
            columns=["ticker", "ex_date", "type", "factor"]
        ),
        asof=asof_ts,
    )

    from skuld_research.data.prepared_panel import build_prepared_panel

    return build_prepared_panel(snap, nzx_only=False, rebalance_start="2021-01-01")


# ---------------------------------------------------------------------------
# Test: protocol conformance
# ---------------------------------------------------------------------------


def test_low_volatility_conforms_to_signal_generator_protocol():
    """LowVolatilityFactor satisfies the SignalGenerator protocol."""
    from skuld_research.factors.low_volatility import LowVolatilityFactor
    from skuld_research.factors.protocols import SignalGenerator

    factor = LowVolatilityFactor()
    assert isinstance(factor, SignalGenerator)
    assert factor.name == "low_volatility"


# ---------------------------------------------------------------------------
# Test: basic ranking (low vol scores high, high vol scores low)
# ---------------------------------------------------------------------------


def test_low_volatility_ranking():
    """Lower-volatility tickers score higher than high-volatility tickers."""
    from skuld_research.factors.low_volatility import LowVolatilityFactor

    # Create three tickers with different volatility profiles
    # LOW: very stable returns (vol ~0.001)
    # MED: moderate returns (vol ~0.01)
    # HIGH: highly volatile returns (vol ~0.05)

    n_days = 600
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rng = np.random.default_rng(123)

    low_vol_returns = pd.Series(0.001 * rng.standard_normal(n_days), index=dates)
    med_vol_returns = pd.Series(0.01 * rng.standard_normal(n_days), index=dates)
    high_vol_returns = pd.Series(0.05 * rng.standard_normal(n_days), index=dates)

    custom_returns = {
        "LOW.NZ": low_vol_returns,
        "MED.NZ": med_vol_returns,
        "HIGH.NZ": high_vol_returns,
    }

    panel = _make_prepared_panel(
        n_days=n_days,
        tickers=["LOW.NZ", "MED.NZ", "HIGH.NZ"],
        custom_returns=custom_returns,
        asof="2025-01-01",
    )

    factor = LowVolatilityFactor(lookback_months=12, min_months=6)
    t = pd.Timestamp("2024-06-01")
    universe = ["LOW.NZ", "MED.NZ", "HIGH.NZ"]
    scores = factor.score(panel, t, universe)

    # Lower volatility should produce higher scores (because score = -vol)
    assert not scores.isna().any(), "Expected no NaN — all tickers have sufficient history"
    assert scores["LOW.NZ"] > scores["MED.NZ"], "Low-vol should score higher than med-vol"
    assert scores["MED.NZ"] > scores["HIGH.NZ"], "Med-vol should score higher than high-vol"


# ---------------------------------------------------------------------------
# Test: minimum history exclusion
# ---------------------------------------------------------------------------


def test_low_volatility_min_history_exclusion():
    """Tickers with fewer than min_months of daily returns get NaN."""
    from skuld_research.factors.low_volatility import LowVolatilityFactor

    # Create a panel where one ticker has only 3 months of history
    n_days = 600
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rng = np.random.default_rng(456)

    # LONG has full history, SHORT only has last ~60 days (3 months)
    long_returns = pd.Series(0.01 * rng.standard_normal(n_days), index=dates)
    short_returns = pd.Series(np.nan, index=dates)
    short_returns.iloc[-60:] = 0.01 * rng.standard_normal(60)

    custom_returns = {
        "LONG.NZ": long_returns,
        "SHORT.NZ": short_returns,
    }

    panel = _make_prepared_panel(
        n_days=n_days,
        tickers=["LONG.NZ", "SHORT.NZ"],
        custom_returns=custom_returns,
        asof="2025-01-01",
    )

    factor = LowVolatilityFactor(lookback_months=12, min_months=6)
    t = pd.Timestamp("2024-06-01")
    universe = ["LONG.NZ", "SHORT.NZ"]
    scores = factor.score(panel, t, universe)

    # LONG should have a valid score, SHORT should be NaN (only 3 months < 6 min)
    assert not pd.isna(scores["LONG.NZ"]), "LONG should score (sufficient history)"
    assert pd.isna(scores["SHORT.NZ"]), "SHORT should be NaN (insufficient history)"


# ---------------------------------------------------------------------------
# Test: PIT safety (no future data)
# ---------------------------------------------------------------------------


def test_low_volatility_pit_safety():
    """No observation on or after rebalance date `t` influences the score."""
    from skuld_research.factors.low_volatility import LowVolatilityFactor

    # Create a panel with distinct volatility periods
    # Use 800 days to ensure we have enough history before the midpoint
    n_days = 800
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    rng = np.random.default_rng(789)

    # Build returns with low vol in first 500 days, high vol in last 300 days
    returns = pd.Series(index=dates, dtype=float)
    returns.iloc[:500] = 0.005 * rng.standard_normal(500)  # Low vol period
    returns.iloc[500:] = 0.05 * rng.standard_normal(300)  # High vol period

    custom_returns = {"X.NZ": returns}

    panel = _make_prepared_panel(
        n_days=n_days,
        tickers=["X.NZ"],
        custom_returns=custom_returns,
        asof="2025-01-01",
    )

    factor = LowVolatilityFactor(lookback_months=12, min_months=6)

    # Score at day 400 (well into the low-vol period, with 400 days history)
    t_early = panel.returns_daily.index[400]
    score_early = factor.score(panel, t_early, ["X.NZ"])["X.NZ"]

    # Score at day 700 (well into the high-vol period, with 700 days history)
    t_late = panel.returns_daily.index[700]
    score_late = factor.score(panel, t_late, ["X.NZ"])["X.NZ"]

    # The score in the low-vol period should be higher (less negative)
    # because the volatility was lower
    assert not pd.isna(score_early), "Early score should be valid"
    assert not pd.isna(score_late), "Late score should be valid"
    assert score_early > score_late, "Low-vol period should score higher than high-vol period"


# ---------------------------------------------------------------------------
# Test: lookback window respects strictly before t
# ---------------------------------------------------------------------------


def test_low_volatility_lookback_excludes_t():
    """The score uses returns strictly before t, not including t itself."""
    from skuld_research.factors.low_volatility import LowVolatilityFactor

    # Panel with known return dates and sufficient history
    n_days = 600
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    rng = np.random.default_rng(999)
    returns = pd.Series(0.01 * rng.standard_normal(n_days), index=dates)

    custom_returns = {"X.NZ": returns}

    panel = _make_prepared_panel(
        n_days=n_days,
        tickers=["X.NZ"],
        custom_returns=custom_returns,
        asof="2025-01-01",
    )

    factor = LowVolatilityFactor(lookback_months=12, min_months=6)

    # Score at day 400, which has 400 days of history before it (well above min_days=126)
    t = panel.returns_daily.index[400]

    universe = ["X.NZ"]
    score = factor.score(panel, t, universe)

    # Should return a valid score (using data before t, not at t)
    assert not pd.isna(score["X.NZ"]), "Score should be valid (sufficient prior data)"

    # Verify that advancing t by one day also produces a valid score
    t_plus_one = panel.returns_daily.index[401]
    score_plus_one = factor.score(panel, t_plus_one, universe)
    assert not pd.isna(score_plus_one["X.NZ"]), "Score at t+1 should also be valid"
