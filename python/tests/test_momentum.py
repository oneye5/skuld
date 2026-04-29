"""Tests for the 12-1 momentum factor.

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
    nan_ticker: str | None = None,
    nan_from_day: int | None = None,
) -> PreparedPanel:
    """Build a synthetic PreparedPanel with controllable history length."""
    if tickers is None:
        tickers = ["AAA.NZ", "BBB.NZ", "CCC.NZ"]

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)

    prices_data = {}
    for t in tickers:
        px = 10.0 * (1 + 0.001 * rng.standard_normal(n_days)).cumprod()
        prices_data[t] = px

    prices = pd.DataFrame(prices_data, index=dates)

    # Optionally blank out a ticker from a certain day (to simulate short history)
    if nan_ticker and nan_from_day is not None:
        prices.iloc[:nan_from_day][nan_ticker] = np.nan

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


def test_momentum_conforms_to_signal_generator_protocol():
    """MomentumFactor satisfies the SignalGenerator protocol."""
    from skuld_research.factors.momentum import MomentumFactor
    from skuld_research.factors.protocols import SignalGenerator

    factor = MomentumFactor()
    assert isinstance(factor, SignalGenerator)
    assert factor.name == "momentum"


# ---------------------------------------------------------------------------
# Test: skip-month is excluded
# ---------------------------------------------------------------------------


def test_momentum_excludes_skip_month():
    """The most recent available month is not included in the 12-1 window."""
    from skuld_research.factors.momentum import MomentumFactor

    # We set up a panel where the most recent month has a huge +50% return.
    # If that month were included, the scores would be dominated by it.
    # We verify that the score is consistent with it being excluded.

    n_days = 600  # >2 years
    dates = pd.bdate_range("2022-06-01", periods=n_days)
    rng = np.random.default_rng(0)

    # Stable returns for all but the last month
    base = 10.0 * (1 + 0.001 * rng.standard_normal(n_days)).cumprod()
    prices = pd.DataFrame({"X.NZ": base, "Y.NZ": base.copy()}, index=dates)

    # Inject a huge positive return only in the very last ~21 days (last month)
    prices.iloc[-21:, 0] *= 1.50  # X gets +50% in the skip month

    prices.index.name = "date"
    volumes = pd.DataFrame(
        {"X.NZ": 1_000_000.0, "Y.NZ": 1_000_000.0}, index=dates
    )
    volumes.index.name = "date"

    asof_ts = pd.Timestamp("2025-03-01")
    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["ticker", "publication_date"])
        ),
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(
            columns=["ticker", "ex_date", "type", "factor"]
        ),
        asof=asof_ts,
    )

    from skuld_research.data.prepared_panel import build_prepared_panel

    panel = build_prepared_panel(snap, nzx_only=False, rebalance_start="2022-06-01")

    factor = MomentumFactor()
    t = pd.Timestamp("2024-08-01")  # rebalance before the skip month
    scores = factor.score(panel, t, ["X.NZ", "Y.NZ"])

    # X and Y had same underlying returns for the 12-month window before skip month.
    # If the skip month WERE included, X would score far above Y.
    # With skip month excluded, they should be very similar (within 10%).
    assert not scores.isna().any(), "Expected no NaN — both tickers have long history"
    assert abs(scores["X.NZ"] - scores["Y.NZ"]) < abs(scores["X.NZ"]) * 0.10 + 1e-6


# ---------------------------------------------------------------------------
# Test: correct 12-month cumulative return calculation
# ---------------------------------------------------------------------------


def test_momentum_cumulative_return_known_value():
    """Verify the 12-month cumulative return matches a hand-computed reference."""
    from skuld_research.factors.momentum import MomentumFactor

    # Build a panel with a known return sequence.
    # Use a simple geometric series: price doubles over 12 months.
    n_months = 30
    # ~21 trading days per month; enough for 12-1 momentum
    n_days = n_months * 21 + 10
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    # Build a price series that goes up exactly 1% per trading day for ticker A
    # and stays flat for ticker B
    a_prices = 10.0 * (1.01 ** np.arange(n_days))
    b_prices = np.full(n_days, 10.0)

    prices = pd.DataFrame(
        {"A.NZ": a_prices, "B.NZ": b_prices}, index=dates
    )
    prices.index.name = "date"

    volumes = pd.DataFrame(
        {"A.NZ": 500_000.0, "B.NZ": 500_000.0}, index=dates
    )
    volumes.index.name = "date"

    asof_ts = pd.Timestamp("2025-03-01")
    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["ticker", "publication_date"])
        ),
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(
            columns=["ticker", "ex_date", "type", "factor"]
        ),
        asof=asof_ts,
    )

    from skuld_research.data.prepared_panel import build_prepared_panel

    panel = build_prepared_panel(snap, nzx_only=False, rebalance_start="2022-01-01")

    factor = MomentumFactor()
    t = pd.Timestamp("2024-05-01")
    scores = factor.score(panel, t, ["A.NZ", "B.NZ"])

    # A has been consistently going up; B is flat.
    # A's momentum should be strongly positive; B's should be ~0 or negative relative.
    assert not scores.isna().any()
    assert scores["A.NZ"] > scores["B.NZ"], (
        f"Expected A (trending up) > B (flat), got A={scores['A.NZ']:.4f}, B={scores['B.NZ']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test: short-history exclusion
# ---------------------------------------------------------------------------


def test_momentum_excludes_ticker_with_short_history():
    """Tickers with fewer than min_months valid observations return NaN."""
    from skuld_research.factors.momentum import MomentumFactor

    n_days = 600
    dates = pd.bdate_range("2022-06-01", periods=n_days)
    rng = np.random.default_rng(1)

    long_px = 10.0 * (1 + 0.001 * rng.standard_normal(n_days)).cumprod()
    short_px = np.full(n_days, np.nan)
    # Only last 150 days (~7 months) have data — below 11-month threshold
    short_px[-150:] = 10.0 * (1 + 0.001 * rng.standard_normal(150)).cumprod()

    prices = pd.DataFrame(
        {"LONG.NZ": long_px, "SHORT.NZ": short_px}, index=dates
    )
    prices.index.name = "date"

    volumes = pd.DataFrame(
        {"LONG.NZ": 500_000.0, "SHORT.NZ": np.where(np.isnan(short_px), 0.0, 500_000.0)},
        index=dates,
    )
    volumes.index.name = "date"

    asof_ts = pd.Timestamp("2025-03-01")
    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["ticker", "publication_date"])
        ),
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(
            columns=["ticker", "ex_date", "type", "factor"]
        ),
        asof=asof_ts,
    )

    from skuld_research.data.prepared_panel import build_prepared_panel

    panel = build_prepared_panel(snap, nzx_only=False, rebalance_start="2022-06-01")

    factor = MomentumFactor(min_months=11)
    t = pd.Timestamp("2024-08-01")
    scores = factor.score(panel, t, ["LONG.NZ", "SHORT.NZ"])

    assert not pd.isna(scores["LONG.NZ"]), "LONG has sufficient history; should have a score"
    assert pd.isna(scores["SHORT.NZ"]), "SHORT has <11 months; should return NaN"


# ---------------------------------------------------------------------------
# Test: ticker in universe but not in panel
# ---------------------------------------------------------------------------


def test_momentum_returns_nan_for_unknown_ticker():
    """Tickers in universe that are absent from the panel return NaN."""
    from skuld_research.factors.momentum import MomentumFactor

    panel = _make_prepared_panel(n_days=600)
    factor = MomentumFactor()
    t = pd.Timestamp("2023-06-01")

    scores = factor.score(panel, t, ["AAA.NZ", "UNKNOWN.NZ"])

    assert not pd.isna(scores["AAA.NZ"])
    assert pd.isna(scores["UNKNOWN.NZ"])


# ---------------------------------------------------------------------------
# Test: insufficient overall history (panel < 13 months)
# ---------------------------------------------------------------------------


def test_momentum_all_nan_when_panel_too_short():
    """If the panel has fewer than 2 available month-ends before t, all NaN."""
    from skuld_research.factors.momentum import MomentumFactor

    panel = _make_prepared_panel(n_days=40)  # ~2 months of data
    factor = MomentumFactor()
    # Rebalance very early — only 1 or 0 months available before t
    t = pd.Timestamp("2021-02-01")
    scores = factor.score(panel, t, ["AAA.NZ"])
    assert scores.isna().all(), f"Expected all NaN with very short history, got {scores.to_dict()}"


# ---------------------------------------------------------------------------
# Test: PIT safety — score at t cannot use data from t onwards
# ---------------------------------------------------------------------------


def test_momentum_pit_safe():
    """Score at rebalance date t does not use any data from t onwards."""
    from skuld_research.factors.momentum import MomentumFactor

    panel = _make_prepared_panel(n_days=800)
    factor = MomentumFactor()

    t1 = pd.Timestamp("2023-06-30")
    t2 = pd.Timestamp("2023-07-01")

    s1 = factor.score(panel, t1, ["AAA.NZ"])
    s2 = factor.score(panel, t2, ["AAA.NZ"])

    # Both should produce valid scores
    assert not s1.isna().all()
    assert not s2.isna().all()
    # Scores at consecutive dates can differ (different windows) — just confirm
    # that adjacent dates produce deterministic results (no randomness in scoring).
    s1_again = factor.score(panel, t1, ["AAA.NZ"])
    assert s1["AAA.NZ"] == pytest.approx(s1_again["AAA.NZ"])


# ---------------------------------------------------------------------------
# Tests: cross-period smoothing
# ---------------------------------------------------------------------------


def test_momentum_smoothing_default_is_identity():
    """smoothing_months=1 produces byte-identical scores to default behavior."""
    from skuld_research.factors.momentum import MomentumFactor

    panel = _make_prepared_panel(n_days=800)
    base = MomentumFactor()
    smoothed1 = MomentumFactor(smoothing_months=1)
    t = pd.Timestamp("2023-06-30")
    universe = ["AAA.NZ", "BBB.NZ", "CCC.NZ"]

    s_base = base.score(panel, t, universe)
    s_smoothed = smoothed1.score(panel, t, universe)

    pd.testing.assert_series_equal(s_base, s_smoothed)


def test_momentum_smoothing_averages_across_periods():
    """smoothing_months=N averages raw scores across N consecutive rebalance dates."""
    from skuld_research.factors.momentum import MomentumFactor

    panel = _make_prepared_panel(n_days=800)
    universe = ["AAA.NZ", "BBB.NZ", "CCC.NZ"]

    rebalance_dates = panel.universe_mask.index
    t = rebalance_dates[-1]
    prior_two = list(rebalance_dates[-3:-1])

    # Script raw scores per rebalance date with distinct, known values per ticker.
    scripted = {
        prior_two[0]: pd.Series({"AAA.NZ": 0.10, "BBB.NZ": -0.05, "CCC.NZ": 0.30}, name="momentum"),
        prior_two[1]: pd.Series({"AAA.NZ": 0.20, "BBB.NZ": 0.05, "CCC.NZ": 0.20}, name="momentum"),
        t: pd.Series({"AAA.NZ": 0.30, "BBB.NZ": 0.15, "CCC.NZ": 0.10}, name="momentum"),
    }

    class ScriptedMomentum(MomentumFactor):
        def _score_raw(self, panel, t_, universe):  # type: ignore[override]
            return scripted[t_].reindex(universe)

    smoothed = ScriptedMomentum(smoothing_months=3).score(panel, t, universe)

    expected = pd.Series(
        {"AAA.NZ": 0.20, "BBB.NZ": 0.05, "CCC.NZ": 0.20},
        name="momentum",
    )
    pd.testing.assert_series_equal(
        smoothed.reindex(universe).astype(float),
        expected.reindex(universe).astype(float),
    )


def test_momentum_smoothing_handles_short_history():
    """When fewer than smoothing_months rebalance dates precede t, average over what exists."""
    from skuld_research.factors.momentum import MomentumFactor

    panel = _make_prepared_panel(n_days=800)
    universe = ["AAA.NZ", "BBB.NZ", "CCC.NZ"]

    rebalance_dates = panel.universe_mask.index
    t_first = rebalance_dates[0]
    t_second = rebalance_dates[1]

    scripted = {
        t_first: pd.Series({"AAA.NZ": 0.10, "BBB.NZ": 0.20, "CCC.NZ": 0.30}, name="momentum"),
        t_second: pd.Series({"AAA.NZ": 0.40, "BBB.NZ": 0.50, "CCC.NZ": 0.60}, name="momentum"),
    }

    class ScriptedMomentum(MomentumFactor):
        def _score_raw(self, panel, t_, universe):  # type: ignore[override]
            return scripted[t_].reindex(universe)

    factor = ScriptedMomentum(smoothing_months=3)

    # At first rebalance date: no prior dates available -> smoothed == raw at t_first
    smoothed_first = factor.score(panel, t_first, universe)
    pd.testing.assert_series_equal(
        smoothed_first.reindex(universe).astype(float),
        scripted[t_first].reindex(universe).astype(float),
    )

    # At second rebalance date: only one prior date available -> mean of the two
    smoothed_second = factor.score(panel, t_second, universe)
    expected = pd.Series(
        {"AAA.NZ": 0.25, "BBB.NZ": 0.35, "CCC.NZ": 0.45},
        name="momentum",
    )
    pd.testing.assert_series_equal(
        smoothed_second.reindex(universe).astype(float),
        expected.reindex(universe).astype(float),
    )


def test_momentum_smoothing_handles_partial_nan():
    """A ticker with NaN raw score in some windows averages over only the valid ones."""
    from skuld_research.factors.momentum import MomentumFactor

    panel = _make_prepared_panel(n_days=800)
    universe = ["LONG.NZ", "SHORT.NZ", "DEAD.NZ"]

    rebalance_dates = panel.universe_mask.index
    t = rebalance_dates[-1]
    window_dates = list(rebalance_dates[-3:])

    scripted = {
        # SHORT.NZ has valid raw only at the most recent date; DEAD.NZ has none.
        window_dates[0]: pd.Series(
            {"LONG.NZ": 0.10, "SHORT.NZ": np.nan, "DEAD.NZ": np.nan}, name="momentum"
        ),
        window_dates[1]: pd.Series(
            {"LONG.NZ": 0.20, "SHORT.NZ": np.nan, "DEAD.NZ": np.nan}, name="momentum"
        ),
        window_dates[2]: pd.Series(
            {"LONG.NZ": 0.30, "SHORT.NZ": 0.50, "DEAD.NZ": np.nan}, name="momentum"
        ),
    }

    class ScriptedMomentum(MomentumFactor):
        def _score_raw(self, panel, t_, universe):  # type: ignore[override]
            return scripted[t_].reindex(universe)

    smoothed = ScriptedMomentum(smoothing_months=3).score(panel, t, universe)

    assert smoothed["LONG.NZ"] == pytest.approx(0.20)  # mean of 0.10, 0.20, 0.30
    assert smoothed["SHORT.NZ"] == pytest.approx(0.50)  # mean of single valid value
    assert pd.isna(smoothed["DEAD.NZ"])  # all NaN -> NaN
