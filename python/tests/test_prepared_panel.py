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


def test_prepared_panel_anomaly_masks_extreme_daily_move_without_volume_confirmation():
    from skuld_research.config.spec import AnomalyFilterSpec

    dates = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame(
        {"SPK.NZ": [10.0, 50.0, 50.0, 50.0]},
        index=dates,
    )
    prices.index.name = "date"
    volumes = pd.DataFrame(
        {"SPK.NZ": [1_000.0, 0.0, 1_000.0, 1_000.0]},
        index=dates,
    )
    volumes.index.name = "date"
    fundamentals = pd.DataFrame(
        {"trailing_basic_average_shares": [1_000_000.0]},
        index=pd.MultiIndex.from_tuples(
            [("SPK.NZ", pd.Timestamp("2024-01-02"))],
            names=["ticker", "publication_date"],
        ),
    )
    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
        asof=pd.Timestamp("2024-02-01", tz="UTC"),
    )

    panel = build_prepared_panel(
        snap,
        anomaly_filter=AnomalyFilterSpec(
            kind="mask_extremes",
            daily_abs_return_threshold=10.0,
            monthly_abs_return_threshold=10.0,
            volume_gate_threshold=0.20,
            require_volume_confirmation=True,
            corporate_action_buffer_days=5,
        ),
    )

    assert pd.isna(panel.prices.loc[dates[1], "SPK.NZ"])
    assert pd.isna(panel.returns_daily.loc[dates[1], "SPK.NZ"])


def test_prepared_panel_anomaly_masks_extreme_monthly_move_without_corporate_action():
    from skuld_research.config.spec import AnomalyFilterSpec

    dates = pd.bdate_range("2024-01-02", "2024-01-31")
    prices = pd.DataFrame(
        {"SPK.NZ": [10.0] * (len(dates) - 1) + [20.0]},
        index=dates,
    )
    prices.index.name = "date"
    volumes = pd.DataFrame(
        {"SPK.NZ": np.full(len(dates), 1_000.0)},
        index=dates,
    )
    volumes.index.name = "date"
    fundamentals = pd.DataFrame(
        {"trailing_basic_average_shares": [1_000_000.0]},
        index=pd.MultiIndex.from_tuples(
            [("SPK.NZ", pd.Timestamp("2024-01-02"))],
            names=["ticker", "publication_date"],
        ),
    )
    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
        asof=pd.Timestamp("2024-02-10", tz="UTC"),
    )

    panel = build_prepared_panel(
        snap,
        anomaly_filter=AnomalyFilterSpec(
            kind="mask_extremes",
            daily_abs_return_threshold=10.0,
            monthly_abs_return_threshold=0.50,
            volume_gate_threshold=10.0,
            require_volume_confirmation=False,
            corporate_action_buffer_days=5,
        ),
    )

    month_end = panel.returns_monthly.index[-1]
    assert panel.prices.loc[dates[-1], "SPK.NZ"] == pytest.approx(20.0)
    assert panel.returns_daily.loc[dates[-1], "SPK.NZ"] == pytest.approx(1.0)
    assert pd.isna(panel.returns_monthly.loc[month_end, "SPK.NZ"])


def test_prepared_panel_anomaly_invalidates_monthly_return_when_bme_label_is_not_bad_price_row():
    from skuld_research.config.spec import AnomalyFilterSpec

    dates = pd.bdate_range("2024-03-01", "2024-03-28")
    prices = pd.DataFrame(
        {"SPK.NZ": [10.0] * (len(dates) - 1) + [20.0]},
        index=dates,
    )
    prices.index.name = "date"
    volumes = pd.DataFrame(
        {"SPK.NZ": np.full(len(dates), 1_000.0)},
        index=dates,
    )
    volumes.index.name = "date"
    fundamentals = pd.DataFrame(
        {"trailing_basic_average_shares": [1_000_000.0]},
        index=pd.MultiIndex.from_tuples(
            [("SPK.NZ", pd.Timestamp("2024-03-01"))],
            names=["ticker", "publication_date"],
        ),
    )
    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
        asof=pd.Timestamp("2024-04-10", tz="UTC"),
    )

    panel = build_prepared_panel(
        snap,
        anomaly_filter=AnomalyFilterSpec(
            kind="mask_extremes",
            daily_abs_return_threshold=10.0,
            monthly_abs_return_threshold=0.50,
            volume_gate_threshold=10.0,
            require_volume_confirmation=False,
            corporate_action_buffer_days=5,
        ),
    )

    month_end = pd.Timestamp("2024-03-29")
    assert panel.prices.loc[pd.Timestamp("2024-03-28"), "SPK.NZ"] == pytest.approx(20.0)
    assert panel.returns_daily.loc[pd.Timestamp("2024-03-28"), "SPK.NZ"] == pytest.approx(1.0)
    assert pd.isna(panel.returns_monthly.loc[month_end, "SPK.NZ"])


def test_prepared_panel_anomaly_keeps_large_reversing_daily_move():
    from skuld_research.config.spec import AnomalyFilterSpec

    dates = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame(
        {"SPK.NZ": [10.0, 30.0, 10.0, 10.0]},
        index=dates,
    )
    prices.index.name = "date"
    volumes = pd.DataFrame(
        {"SPK.NZ": [1_000.0, 1_000.0, 1_000.0, 1_000.0]},
        index=dates,
    )
    volumes.index.name = "date"
    fundamentals = pd.DataFrame(
        {"trailing_basic_average_shares": [1_000_000.0]},
        index=pd.MultiIndex.from_tuples(
            [("SPK.NZ", pd.Timestamp("2024-01-02"))],
            names=["ticker", "publication_date"],
        ),
    )
    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
        asof=pd.Timestamp("2024-02-01", tz="UTC"),
    )

    panel = build_prepared_panel(
        snap,
        anomaly_filter=AnomalyFilterSpec(
            kind="mask_extremes",
            daily_abs_return_threshold=1.0,
            monthly_abs_return_threshold=10.0,
            volume_gate_threshold=10.0,
            require_volume_confirmation=False,
            corporate_action_buffer_days=5,
        ),
    )

    assert panel.prices.loc[dates[1], "SPK.NZ"] == pytest.approx(30.0)
    assert panel.returns_daily.loc[dates[1], "SPK.NZ"] == pytest.approx(2.0)
    assert panel.returns_daily.loc[dates[2], "SPK.NZ"] == pytest.approx(-2.0 / 3.0)


def test_prepared_panel_anomaly_masks_intraday_prices_after_daily_normalization():
    from skuld_research.config.spec import AnomalyFilterSpec

    dates = pd.to_datetime(
        [
            "2024-01-02 10:00:00",
            "2024-01-02 16:00:00",
            "2024-01-03 10:00:00",
            "2024-01-03 16:00:00",
            "2024-01-04 10:00:00",
            "2024-01-04 16:00:00",
        ]
    )
    prices = pd.DataFrame(
        {"SPK.NZ": [10.0, 10.0, 50.0, 50.0, 50.0, 50.0]},
        index=dates,
    )
    prices.index.name = "date"
    volumes = pd.DataFrame(
        {"SPK.NZ": [1_000.0, 1_000.0, 0.0, 0.0, 1_000.0, 1_000.0]},
        index=dates,
    )
    volumes.index.name = "date"
    fundamentals = pd.DataFrame(
        {"trailing_basic_average_shares": [1_000_000.0]},
        index=pd.MultiIndex.from_tuples(
            [("SPK.NZ", pd.Timestamp("2024-01-02"))],
            names=["ticker", "publication_date"],
        ),
    )
    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
        asof=pd.Timestamp("2024-02-01", tz="UTC"),
    )

    panel = build_prepared_panel(
        snap,
        anomaly_filter=AnomalyFilterSpec(
            kind="mask_extremes",
            daily_abs_return_threshold=10.0,
            monthly_abs_return_threshold=10.0,
            volume_gate_threshold=0.20,
            require_volume_confirmation=True,
            corporate_action_buffer_days=5,
        ),
    )

    assert pd.isna(panel.prices.loc[pd.Timestamp("2024-01-03"), "SPK.NZ"])
    assert pd.isna(panel.returns_daily.loc[pd.Timestamp("2024-01-03"), "SPK.NZ"])


def test_prepared_panel_anomaly_does_not_auto_fail_final_date_volume_confirmation():
    from skuld_research.config.spec import AnomalyFilterSpec

    dates = pd.bdate_range("2024-01-02", periods=2)
    prices = pd.DataFrame({"SPK.NZ": [10.0, 50.0]}, index=dates)
    prices.index.name = "date"
    volumes = pd.DataFrame({"SPK.NZ": [1_000.0, 1_000.0]}, index=dates)
    volumes.index.name = "date"
    fundamentals = pd.DataFrame(
        {"trailing_basic_average_shares": [1_000_000.0]},
        index=pd.MultiIndex.from_tuples(
            [("SPK.NZ", pd.Timestamp("2024-01-02"))],
            names=["ticker", "publication_date"],
        ),
    )
    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
        asof=pd.Timestamp("2024-02-01", tz="UTC"),
    )

    panel = build_prepared_panel(
        snap,
        anomaly_filter=AnomalyFilterSpec(
            kind="mask_extremes",
            daily_abs_return_threshold=10.0,
            monthly_abs_return_threshold=10.0,
            volume_gate_threshold=0.20,
            require_volume_confirmation=True,
            corporate_action_buffer_days=5,
        ),
    )

    assert panel.prices.loc[dates[-1], "SPK.NZ"] == pytest.approx(50.0)
    assert panel.returns_daily.loc[dates[-1], "SPK.NZ"] == pytest.approx(4.0)


def test_prepared_panel_anomaly_monthly_corporate_action_uses_offending_trading_date():
    from skuld_research.config.spec import AnomalyFilterSpec

    dates = pd.bdate_range("2024-03-01", "2024-03-28")
    prices = pd.DataFrame(
        {"SPK.NZ": [10.0] * (len(dates) - 1) + [20.0]},
        index=dates,
    )
    prices.index.name = "date"
    volumes = pd.DataFrame(
        {"SPK.NZ": np.full(len(dates), 1_000.0)},
        index=dates,
    )
    volumes.index.name = "date"
    fundamentals = pd.DataFrame(
        {"trailing_basic_average_shares": [1_000_000.0]},
        index=pd.MultiIndex.from_tuples(
            [("SPK.NZ", pd.Timestamp("2024-03-01"))],
            names=["ticker", "publication_date"],
        ),
    )
    corporate_actions = pd.DataFrame(
        {
            "ticker": ["SPK.NZ"],
            "ex_date": [pd.Timestamp("2024-03-28")],
            "type": ["split"],
            "factor": [2.0],
        }
    )
    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        macro=pd.DataFrame(),
        corporate_actions=corporate_actions,
        asof=pd.Timestamp("2024-04-10", tz="UTC"),
    )

    panel = build_prepared_panel(
        snap,
        anomaly_filter=AnomalyFilterSpec(
            kind="mask_extremes",
            daily_abs_return_threshold=10.0,
            monthly_abs_return_threshold=0.50,
            volume_gate_threshold=10.0,
            require_volume_confirmation=False,
            corporate_action_buffer_days=0,
        ),
    )

    assert panel.returns_monthly.loc[pd.Timestamp("2024-03-29"), "SPK.NZ"] == pytest.approx(1.0)


def test_prepared_panel_anomaly_invalidates_extreme_monthly_return_with_mixed_daily_signs():
    from skuld_research.config.spec import AnomalyFilterSpec

    dates = pd.bdate_range("2024-01-02", periods=6)
    prices = pd.DataFrame(
        {"SPK.NZ": [10.0, 30.0, 20.0, 80.0, 70.0, 80.0]},
        index=dates,
    )
    prices.index.name = "date"
    volumes = pd.DataFrame({"SPK.NZ": np.full(len(dates), 1_000.0)}, index=dates)
    volumes.index.name = "date"
    fundamentals = pd.DataFrame(
        {"trailing_basic_average_shares": [1_000_000.0]},
        index=pd.MultiIndex.from_tuples(
            [("SPK.NZ", pd.Timestamp("2024-01-02"))],
            names=["ticker", "publication_date"],
        ),
    )
    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
        asof=pd.Timestamp("2024-02-10", tz="UTC"),
    )

    panel = build_prepared_panel(
        snap,
        anomaly_filter=AnomalyFilterSpec(
            kind="mask_extremes",
            daily_abs_return_threshold=10.0,
            monthly_abs_return_threshold=5.0,
            volume_gate_threshold=10.0,
            require_volume_confirmation=False,
            corporate_action_buffer_days=5,
        ),
    )

    month_end = panel.returns_monthly.index[-1]
    assert pd.isna(panel.returns_monthly.loc[month_end, "SPK.NZ"])


def test_prepared_panel_volume_gate_respects_nearby_corporate_action():
    from skuld_research.config.spec import AnomalyFilterSpec

    dates = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame({"SPK.NZ": [10.0, 50.0, 50.0, 50.0]}, index=dates)
    prices.index.name = "date"
    volumes = pd.DataFrame({"SPK.NZ": [1_000.0, 0.0, 1_000.0, 1_000.0]}, index=dates)
    volumes.index.name = "date"
    fundamentals = pd.DataFrame(
        {"trailing_basic_average_shares": [1_000_000.0]},
        index=pd.MultiIndex.from_tuples(
            [("SPK.NZ", pd.Timestamp("2024-01-02"))],
            names=["ticker", "publication_date"],
        ),
    )
    corporate_actions = pd.DataFrame(
        {"ticker": ["SPK.NZ"], "ex_date": [dates[1]], "type": ["split"], "factor": [5.0]}
    )
    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        macro=pd.DataFrame(),
        corporate_actions=corporate_actions,
        asof=pd.Timestamp("2024-02-10", tz="UTC"),
    )

    panel = build_prepared_panel(
        snap,
        anomaly_filter=AnomalyFilterSpec(
            kind="mask_extremes",
            daily_abs_return_threshold=10.0,
            monthly_abs_return_threshold=10.0,
            volume_gate_threshold=0.20,
            require_volume_confirmation=True,
            corporate_action_buffer_days=0,
        ),
    )

    assert panel.prices.loc[dates[1], "SPK.NZ"] == pytest.approx(50.0)
    assert panel.returns_daily.loc[dates[1], "SPK.NZ"] == pytest.approx(4.0)


def test_market_cap_proxy_present_and_covers_gaps():
    """market_cap_proxy should cover pre-publication dates that market_cap misses.

    The synthetic fixture has prices starting 2024-01-02 but only one fundamental
    publication at 2024-03-01. market_cap is NaN for all dates before that publication.
    market_cap_proxy should back-fill to cover those pre-publication dates, giving it
    strictly more non-NaN coverage.
    """
    snap = _make_snap(pd.Timestamp("2025-01-01", tz="UTC"))
    panel = build_prepared_panel(snap)

    assert not panel.market_cap_proxy.empty, "market_cap_proxy should not be empty"

    # Proxy must cover strictly more values: it back-fills pre-publication dates
    mc_valid = panel.market_cap.notna().sum().sum()
    proxy_valid = panel.market_cap_proxy.notna().sum().sum()
    assert proxy_valid > mc_valid, (
        f"market_cap_proxy ({proxy_valid}) should exceed market_cap ({mc_valid}) coverage "
        "because proxy back-fills dates before the first fundamental publication"
    )

    # Verify pre-publication trading dates are NaN in market_cap but non-NaN in proxy
    pub_date = pd.Timestamp("2024-03-01")
    pre_pub = panel.market_cap.index[panel.market_cap.index < pub_date]
    assert len(pre_pub) > 0, "fixture must have dates before publication"
    # market_cap is NaN pre-publication (shares not yet published)
    pre_pub_mc = panel.market_cap.loc[pre_pub]
    assert pre_pub_mc.isna().all().all(), "market_cap should be NaN pre-publication"
    # proxy is non-NaN wherever prices are non-NaN (bfill provides share count)
    pre_pub_proxy = panel.market_cap_proxy.loc[pre_pub]
    prices_pre_pub = panel.prices.loc[pre_pub]
    price_available = prices_pre_pub.notna()
    proxy_available = pre_pub_proxy.notna()
    assert (proxy_available | ~price_available).all().all(), (
        "market_cap_proxy should be non-NaN wherever prices are available pre-publication"
    )

    # Magnitudes should be plausible (proxy and market_cap agree post-publication)
    post_pub = panel.market_cap.index[panel.market_cap.index >= pub_date]
    if len(post_pub) > 0:
        mc_post = panel.market_cap.loc[post_pub].stack()
        proxy_post = panel.market_cap_proxy.loc[post_pub].stack()
        # Both should be in the same ballpark (ratio close to 1)
        ratio = (proxy_post / mc_post).dropna()
        assert (ratio - 1.0).abs().max() < 1e-6, "proxy and market_cap should agree post-publication"


def test_market_cap_proxy_no_fundamentals():
    """market_cap_proxy should be all-NaN (not error) when fundamentals are absent."""
    dates = pd.bdate_range("2024-01-02", periods=50)
    prices = pd.DataFrame({"ANZ.NZ": [50.0] * 50}, index=dates)
    prices.index.name = "date"
    volumes = pd.DataFrame({"ANZ.NZ": [500_000.0] * 50}, index=dates)
    volumes.index.name = "date"

    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=pd.DataFrame(),
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
        asof=pd.Timestamp("2025-01-01"),
    )
    panel = build_prepared_panel(snap)
    # proxy should exist but be all NaN
    assert panel.market_cap_proxy.notna().sum().sum() == 0
