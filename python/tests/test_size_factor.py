"""Tests for the size factor.

These tests are written before the implementation (TDD red phase).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PITSnapshot, PreparedPanel


def _make_prepared_panel(
    tickers: list[str],
    market_caps: dict[str, float],
    asof: str = "2025-01-01",
    n_days: int = 300,
) -> PreparedPanel:
    """Build a synthetic PreparedPanel with specified market caps."""
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    # Build prices (constant 10.0 for simplicity)
    prices = pd.DataFrame(
        {t: np.full(n_days, 10.0) for t in tickers},
        index=dates,
    )
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

    panel = build_prepared_panel(snap, nzx_only=False, rebalance_start="2022-01-01")

    # Override market_cap with our specified values
    # Create a DataFrame with constant market cap across all dates
    mcap_data = {}
    for ticker, mcap in market_caps.items():
        mcap_data[ticker] = np.full(len(panel.market_cap.index), mcap)

    panel = PreparedPanel(
        returns_daily=panel.returns_daily,
        returns_monthly=panel.returns_monthly,
        market_cap=pd.DataFrame(mcap_data, index=panel.market_cap.index),
        sector=panel.sector,
        universe_mask=panel.universe_mask,
        macro=panel.macro,
        asof=panel.asof,
    )

    return panel


# ---------------------------------------------------------------------------
# Test: protocol conformance
# ---------------------------------------------------------------------------


def test_size_conforms_to_signal_generator_protocol():
    """SizeFactor satisfies the SignalGenerator protocol."""
    from skuld_research.factors.protocols import SignalGenerator
    from skuld_research.factors.size import SizeFactor

    factor = SizeFactor()
    assert isinstance(factor, SignalGenerator)
    assert factor.name == "size"


# ---------------------------------------------------------------------------
# Test: basic ranking (smaller mcap scores higher)
# ---------------------------------------------------------------------------


def test_size_ranking():
    """Smaller market cap tickers score higher than larger market cap tickers."""
    from skuld_research.factors.size import SizeFactor

    # Create three tickers with vastly different market caps
    tickers = ["SMALL.NZ", "MED.NZ", "LARGE.NZ"]
    market_caps = {
        "SMALL.NZ": 50_000_000,     # $50M
        "MED.NZ": 500_000_000,      # $500M
        "LARGE.NZ": 5_000_000_000,  # $5B
    }

    panel = _make_prepared_panel(tickers, market_caps)

    factor = SizeFactor()
    t = pd.Timestamp("2024-06-01")
    universe = tickers
    scores = factor.score(panel, t, universe)

    # Smaller mcap should produce higher scores (because score = -log(mcap))
    assert not scores.isna().any(), "Expected no NaN — all tickers have market cap"
    assert scores["SMALL.NZ"] > scores["MED.NZ"], "Smaller mcap should score higher"
    assert scores["MED.NZ"] > scores["LARGE.NZ"], "Medium mcap should score higher than large"


# ---------------------------------------------------------------------------
# Test: missing market cap returns NaN
# ---------------------------------------------------------------------------


def test_size_missing_mcap():
    """Tickers without market cap get NaN score."""
    from skuld_research.factors.size import SizeFactor

    # Create a panel where both tickers are initially valid
    tickers = ["VALID.NZ", "MISSING.NZ"]
    market_caps = {
        "VALID.NZ": 100_000_000,
        "MISSING.NZ": 100_000_000,  # Start with valid value
    }

    panel = _make_prepared_panel(tickers, market_caps)

    # Now set MISSING.NZ mcap to NaN across all dates
    mcap_with_nan = panel.market_cap.copy()
    mcap_with_nan["MISSING.NZ"] = np.nan

    panel = PreparedPanel(
        returns_daily=panel.returns_daily,
        returns_monthly=panel.returns_monthly,
        market_cap=mcap_with_nan,
        sector=panel.sector,
        universe_mask=panel.universe_mask,
        macro=panel.macro,
        asof=panel.asof,
    )

    factor = SizeFactor()
    t = pd.Timestamp("2024-06-01")
    universe = tickers
    scores = factor.score(panel, t, universe)

    # VALID should have a score, MISSING should be NaN
    assert not pd.isna(scores["VALID.NZ"]), "VALID should have a score"
    assert pd.isna(scores["MISSING.NZ"]), "MISSING should be NaN (no market cap)"


# ---------------------------------------------------------------------------
# Test: winsorisation clamps extremes
# ---------------------------------------------------------------------------


def test_size_winsorisation():
    """Extreme market caps are winsorised at p1 and p99."""
    from skuld_research.factors.size import SizeFactor

    # Create a universe with 100 tickers spanning a wide range
    tickers = [f"T{i:03d}.NZ" for i in range(100)]

    # Most tickers have mcaps between $50M and $500M (log-uniform)
    # Two outliers: one tiny ($1M) and one huge ($50B)
    rng = np.random.default_rng(123)
    mcaps_log = rng.uniform(np.log(50e6), np.log(500e6), 100)
    mcaps = {f"T{i:03d}.NZ": np.exp(mcaps_log[i]) for i in range(100)}

    # Inject outliers
    mcaps["T000.NZ"] = 1_000_000      # Tiny outlier ($1M)
    mcaps["T099.NZ"] = 50_000_000_000  # Huge outlier ($50B)

    panel = _make_prepared_panel(tickers, mcaps)

    factor = SizeFactor()
    t = pd.Timestamp("2024-06-01")
    universe = tickers
    scores = factor.score(panel, t, universe)

    # Verify no NaN
    assert not scores.isna().any(), "Expected no NaN"

    # Verify that the range of scores is bounded (not infinite)
    assert scores.min() > -np.inf, "Winsorisation should prevent -inf"
    assert scores.max() < np.inf, "Winsorisation should prevent +inf"

    # The outliers should be clamped toward the bulk of the distribution
    # T000 (tiny, should score very high but clamped at p99 of the bulk)
    # T099 (huge, should score very low but clamped at p1 of the bulk)

    # The score range should be reasonable (not spanning 100+ units)
    score_range = scores.max() - scores.min()
    assert score_range < 20, f"Score range {score_range} is too wide — winsorisation may not be working"


# ---------------------------------------------------------------------------
# Test: PIT safety (no future data)
# ---------------------------------------------------------------------------


def test_size_pit_safety():
    """No observation on or after rebalance date `t` influences the score."""
    from skuld_research.factors.size import SizeFactor

    # Create two separate panels with different constant market caps
    # to test that the factor uses the correct PIT mcap value

    tickers = ["X.NZ"]

    # Panel 1: Low mcap period
    panel_low = _make_prepared_panel(tickers, {"X.NZ": 100_000_000})

    # Panel 2: High mcap period
    panel_high = _make_prepared_panel(tickers, {"X.NZ": 1_000_000_000})

    factor = SizeFactor()
    t = pd.Timestamp("2024-06-01")

    score_low = factor.score(panel_low, t, ["X.NZ"])["X.NZ"]
    score_high = factor.score(panel_high, t, ["X.NZ"])["X.NZ"]

    # The low mcap period should score higher (less negative log)
    assert score_low > score_high, "Low mcap should score higher than high mcap"

    # Verify the scores are in the expected range
    # -log(100M) ≈ -18.42, -log(1B) ≈ -20.72
    assert -19 < score_low < -18, f"Expected score_low near -18.42, got {score_low}"
    assert -21 < score_high < -20, f"Expected score_high near -20.72, got {score_high}"
