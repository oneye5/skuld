"""Tests for PreparedPanel builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import PITSnapshot
from skuld_research.data.prepared_panel import build_prepared_panel


def _make_snap(
    asof: pd.Timestamp,
    *,
    spk_volume: float = 100.0,
) -> PITSnapshot:
    """Synthetic snapshot: 250 business days, 2 tickers, 1 fundamental row each."""
    dates = pd.bdate_range("2024-01-02", periods=250)
    rng = np.random.default_rng(0)
    anz = 50.0 * (1 + 0.001 * rng.standard_normal(250)).cumprod()
    spk = 4.80 * (1 + 0.005 * rng.standard_normal(250)).cumprod()
    prices = pd.DataFrame({"ANZ.NZ": anz, "SPK.NZ": spk}, index=dates)
    prices.index.name = "date"

    volumes = pd.DataFrame(
        {"ANZ.NZ": np.full(250, 500_000.0), "SPK.NZ": np.full(250, spk_volume)},
        index=dates,
    )
    volumes.index.name = "date"

    fund_idx = pd.MultiIndex.from_tuples(
        [
            ("ANZ.NZ", pd.Timestamp("2024-03-01")),
            ("SPK.NZ", pd.Timestamp("2024-03-01")),
        ],
        names=["ticker", "publication_date"],
    )
    fundamentals = pd.DataFrame(
        {"trailing_basic_average_shares": [3_000_000_000.0, 1_870_000_000.0]},
        index=fund_idx,
    )

    return PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
        asof=asof,
    )


def test_returns_daily_from_pct_change():
    snap = _make_snap(pd.Timestamp("2025-01-01", tz="UTC"))
    panel = build_prepared_panel(snap)
    expected_first = snap.prices["ANZ.NZ"].iloc[1] / snap.prices["ANZ.NZ"].iloc[0] - 1
    assert panel.returns_daily["ANZ.NZ"].iloc[1] == pytest.approx(expected_first)
    assert pd.isna(panel.returns_daily["ANZ.NZ"].iloc[0])


def test_returns_monthly_is_month_end():
    snap = _make_snap(pd.Timestamp("2025-01-01", tz="UTC"))
    panel = build_prepared_panel(snap)
    for d in panel.returns_monthly.index:
        next_day = d + pd.offsets.BDay(1)
        assert next_day.month != d.month, f"{d} is not month-end"


def test_market_cap_uses_share_fallback():
    snap = _make_snap(pd.Timestamp("2025-01-01", tz="UTC"))
    panel = build_prepared_panel(snap)
    last_date = panel.market_cap.index[-1]
    assert panel.market_cap.loc[last_date, "ANZ.NZ"] == pytest.approx(
        snap.prices.loc[last_date, "ANZ.NZ"] * 3_000_000_000
    )


def test_sector_is_unknown_for_all():
    snap = _make_snap(pd.Timestamp("2025-01-01", tz="UTC"))
    panel = build_prepared_panel(snap)
    assert (panel.sector == "Unknown").all()
    assert set(panel.sector.index) == set(snap.prices.columns)


def test_universe_mask_excludes_low_liquidity():
    snap = _make_snap(pd.Timestamp("2025-01-01", tz="UTC"), spk_volume=100.0)
    panel = build_prepared_panel(snap)
    # SPK volume = 100 shares/day @ ~$5 = $500/day << $10k threshold -> excluded
    assert not panel.universe_mask["SPK.NZ"].any(), (
        "Low-liquidity ticker should be excluded from universe"
    )
    assert panel.universe_mask["ANZ.NZ"].any()


def test_universe_mask_excludes_short_history():
    asof = pd.Timestamp("2025-01-01", tz="UTC")
    dates = pd.bdate_range("2024-01-02", periods=250)
    prices = pd.DataFrame(
        {"ANZ.NZ": np.full(250, 50.0), "NEW.NZ": np.full(250, 10.0)},
        index=dates,
    )
    prices.index.name = "date"
    prices.iloc[:200, prices.columns.get_loc("NEW.NZ")] = np.nan
    volumes = pd.DataFrame(
        {"ANZ.NZ": np.full(250, 500_000.0), "NEW.NZ": np.full(250, 500_000.0)},
        index=dates,
    )
    volumes.index.name = "date"
    fund_idx = pd.MultiIndex.from_tuples(
        [
            ("ANZ.NZ", pd.Timestamp("2024-03-01")),
            ("NEW.NZ", pd.Timestamp("2024-12-01")),
        ],
        names=["ticker", "publication_date"],
    )
    fundamentals = pd.DataFrame(
        {"trailing_basic_average_shares": [3_000_000_000.0, 500_000_000.0]},
        index=fund_idx,
    )
    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
        asof=asof,
    )
    panel = build_prepared_panel(snap)
    assert not panel.universe_mask["NEW.NZ"].any()


def test_rebalance_start_clips_early_dates():
    """rebalance_start excludes month-ends before the given cutoff."""
    snap = _make_snap(pd.Timestamp("2025-01-01", tz="UTC"))
    panel_default = build_prepared_panel(snap)
    # Default: first date with 2+ tickers in data (min(10, 2) = 2), which for
    # this 2024-only synthetic snap is 2024-01-02, so first rebalance is Jan 2024.
    assert panel_default.universe_mask.index.min() >= pd.Timestamp("2024-01-01")

    # Explicit cutoff: only include rebalance dates from October 2024 onward.
    panel_late = build_prepared_panel(snap, rebalance_start="2024-10-01")
    assert panel_late.universe_mask.index.min() >= pd.Timestamp("2024-10-01")
    assert len(panel_late.universe_mask) < len(panel_default.universe_mask)


def test_panel_asof_propagates():
    asof = pd.Timestamp("2025-01-01", tz="UTC")
    panel = build_prepared_panel(_make_snap(asof))
    assert panel.asof == asof


def test_returns_daily_rows_strictly_before_asof():
    asof = pd.Timestamp("2025-01-01", tz="UTC")
    panel = build_prepared_panel(_make_snap(asof))
    asof_naive = asof.tz_localize(None)
    assert panel.returns_daily.index.max() < asof_naive


def test_nzx_only_drops_non_nz_tickers_by_default():
    """Non-`.NZ` tickers (FX/macro indices, futures) are excluded by default."""
    asof = pd.Timestamp("2025-01-01", tz="UTC")
    dates = pd.bdate_range("2024-01-02", periods=250)
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        {
            "ANZ.NZ": 50.0 * (1 + 0.001 * rng.standard_normal(250)).cumprod(),
            "%5ETNX": 4.5 * (1 + 0.001 * rng.standard_normal(250)).cumprod(),
            "ZS=F": 1100.0 * (1 + 0.001 * rng.standard_normal(250)).cumprod(),
        },
        index=dates,
    )
    prices.index.name = "date"
    volumes = pd.DataFrame(
        {
            "ANZ.NZ": np.full(250, 500_000.0),
            "%5ETNX": np.full(250, 0.0),
            "ZS=F": np.full(250, 0.0),
        },
        index=dates,
    )
    volumes.index.name = "date"
    fund_idx = pd.MultiIndex.from_tuples(
        [("ANZ.NZ", pd.Timestamp("2024-03-01"))],
        names=["ticker", "publication_date"],
    )
    fundamentals = pd.DataFrame(
        {"trailing_basic_average_shares": [3_000_000_000.0]},
        index=fund_idx,
    )
    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
        asof=asof,
    )

    panel = build_prepared_panel(snap)
    assert list(panel.returns_daily.columns) == ["ANZ.NZ"]
    assert list(panel.universe_mask.columns) == ["ANZ.NZ"]
    assert "%5ETNX" not in panel.market_cap.columns
    assert "ZS=F" not in panel.market_cap.columns

    # Opt-in: keep them when explicitly requested.
    panel_all = build_prepared_panel(snap, nzx_only=False)
    assert "%5ETNX" in panel_all.returns_daily.columns
    assert "ZS=F" in panel_all.returns_daily.columns


def _make_long_snap(asof: pd.Timestamp, *, n_days: int = 600) -> PITSnapshot:
    """Synthetic snapshot with a longer history (default ~2.5y) for cadence tests."""
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rng = np.random.default_rng(7)
    anz = 50.0 * (1 + 0.001 * rng.standard_normal(n_days)).cumprod()
    spk = 4.80 * (1 + 0.005 * rng.standard_normal(n_days)).cumprod()
    prices = pd.DataFrame({"ANZ.NZ": anz, "SPK.NZ": spk}, index=dates)
    prices.index.name = "date"
    volumes = pd.DataFrame(
        {"ANZ.NZ": np.full(n_days, 500_000.0), "SPK.NZ": np.full(n_days, 500_000.0)},
        index=dates,
    )
    volumes.index.name = "date"
    fund_idx = pd.MultiIndex.from_tuples(
        [
            ("ANZ.NZ", pd.Timestamp("2023-03-01")),
            ("SPK.NZ", pd.Timestamp("2023-03-01")),
        ],
        names=["ticker", "publication_date"],
    )
    fundamentals = pd.DataFrame(
        {"trailing_basic_average_shares": [3_000_000_000.0, 1_870_000_000.0]},
        index=fund_idx,
    )
    return PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
        asof=asof,
    )


def test_prepared_panel_rebalance_freq_default_unchanged():
    """Default rebalance cadence is business month-end."""
    snap = _make_long_snap(pd.Timestamp("2025-06-01", tz="UTC"))
    panel = build_prepared_panel(snap)
    # Every rebal date is a month-end (next business day is in a different month).
    for d in panel.universe_mask.index:
        next_day = d + pd.offsets.BDay(1)
        assert next_day.month != d.month, f"{d} is not a business month-end"


def test_prepared_panel_rebalance_freq_quarterly():
    """rebalance_freq='BQE' yields quarter-end rebal dates and ~1/3 the count."""
    asof = pd.Timestamp("2025-06-01", tz="UTC")
    snap = _make_long_snap(asof)
    panel_m = build_prepared_panel(snap)
    panel_q = build_prepared_panel(snap, rebalance_freq="BQE")

    # Quarterly count should be roughly a third of monthly (allow loose bounds).
    assert len(panel_q.universe_mask) <= len(panel_m.universe_mask) // 2
    assert len(panel_q.universe_mask) >= 5  # ~2 years of quarters

    # Each quarterly rebal date is a quarter-end month (Mar/Jun/Sep/Dec).
    for d in panel_q.universe_mask.index:
        assert d.month in {3, 6, 9, 12}, f"{d} is not a quarter-end month"
        next_day = d + pd.offsets.BDay(1)
        assert next_day.month != d.month, f"{d} is not a business month-end"


def test_prepared_panel_rebalance_freq_invalid_raises():
    snap = _make_long_snap(pd.Timestamp("2025-06-01", tz="UTC"))
    with pytest.raises(ValueError, match="rebalance_freq"):
        build_prepared_panel(snap, rebalance_freq="BWE")


def test_prepared_panel_rebalance_freq_returns_monthly_unchanged():
    """returns_monthly is always BME-aggregated regardless of rebal cadence."""
    snap = _make_long_snap(pd.Timestamp("2025-06-01", tz="UTC"))
    panel_m = build_prepared_panel(snap)
    panel_q = build_prepared_panel(snap, rebalance_freq="BQE")
    pd.testing.assert_frame_equal(panel_m.returns_monthly, panel_q.returns_monthly)
