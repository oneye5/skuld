"""Tests for PITSnapshot and PreparedPanel contract types."""

import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import (
    BacktestResult,
    CombinedScores,
    CurrentPortfolio,
    FoldResult,
    PITSnapshot,
    PreparedPanel,
    TargetPortfolio,
    TradeList,
    WalkForwardResult,
)


def test_pit_snapshot_construction():
    """PITSnapshot can be constructed with valid DataFrames."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    snap = PITSnapshot(
        prices=pd.DataFrame(
            {"ANZ.NZ": [50.0, 51.0]},
            index=pd.to_datetime(["2025-01-13", "2025-01-14"]),
        ),
        volumes=pd.DataFrame(
            {"ANZ.NZ": [100_000.0, 120_000.0]},
            index=pd.to_datetime(["2025-01-13", "2025-01-14"]),
        ),
        fundamentals=pd.DataFrame(
            {"annual_net_income_common_stockholders": [1_000_000.0]},
            index=pd.MultiIndex.from_tuples(
                [("ANZ.NZ", pd.Timestamp("2024-06-30"))],
                names=["ticker", "publication_date"],
            ),
        ),
        macro=pd.DataFrame(
            {"oecd_bcicp": [100.5]},
            index=pd.to_datetime(["2025-01-10"]),
        ),
        corporate_actions=pd.DataFrame(
            {
                "ticker": ["ANZ.NZ"],
                "ex_date": [pd.Timestamp("2024-12-01")],
                "type": ["dividend"],
                "factor": [0.50],
            }
        ),
        asof=ts,
    )
    assert snap.asof == ts
    assert list(snap.prices.columns) == ["ANZ.NZ"]
    assert snap.prices.shape == (2, 1)


def test_pit_snapshot_rejects_future_prices():
    """PITSnapshot raises if prices contain dates >= asof."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    future_prices = pd.DataFrame(
        {"ANZ.NZ": [50.0, 51.0, 52.0]},
        index=pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"]),
    )
    try:
        PITSnapshot(
            prices=future_prices,
            volumes=pd.DataFrame(),
            fundamentals=pd.DataFrame(),
            macro=pd.DataFrame(),
            corporate_actions=pd.DataFrame(),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "future" in str(e).lower() or "asof" in str(e).lower()


def test_pit_snapshot_rejects_future_volumes():
    """PITSnapshot raises if volumes contain dates >= asof."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    try:
        PITSnapshot(
            prices=pd.DataFrame(),
            volumes=pd.DataFrame(
                {"ANZ.NZ": [100_000.0, 120_000.0, 130_000.0]},
                index=pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"]),
            ),
            fundamentals=pd.DataFrame(),
            macro=pd.DataFrame(),
            corporate_actions=pd.DataFrame(),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "volumes" in str(e).lower()


def test_pit_snapshot_rejects_future_macro():
    """PITSnapshot raises if macro contains dates >= asof."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    try:
        PITSnapshot(
            prices=pd.DataFrame(),
            volumes=pd.DataFrame(),
            fundamentals=pd.DataFrame(),
            macro=pd.DataFrame(
                {"oecd_bcicp": [100.5, 101.0]},
                index=pd.to_datetime(["2025-01-14", "2025-01-15"]),
            ),
            corporate_actions=pd.DataFrame(),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "macro" in str(e).lower()


def test_pit_snapshot_rejects_future_fundamentals():
    """PITSnapshot raises if fundamentals have publication_date >= asof."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    try:
        PITSnapshot(
            prices=pd.DataFrame(),
            volumes=pd.DataFrame(),
            fundamentals=pd.DataFrame(
                {"annual_net_income_common_stockholders": [1_000_000.0]},
                index=pd.MultiIndex.from_tuples(
                    [("ANZ.NZ", pd.Timestamp("2025-01-15"))],
                    names=["ticker", "publication_date"],
                ),
            ),
            macro=pd.DataFrame(),
            corporate_actions=pd.DataFrame(),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "fundamentals" in str(e).lower()


def test_pit_snapshot_rejects_future_corporate_actions():
    """PITSnapshot raises if corporate_actions have ex_date >= asof."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    try:
        PITSnapshot(
            prices=pd.DataFrame(),
            volumes=pd.DataFrame(),
            fundamentals=pd.DataFrame(),
            macro=pd.DataFrame(),
            corporate_actions=pd.DataFrame(
                {
                    "ticker": ["ANZ.NZ"],
                    "ex_date": [pd.Timestamp("2025-01-15")],
                    "type": ["dividend"],
                    "factor": [0.50],
                }
            ),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "corporate_actions" in str(e).lower()


def test_pit_snapshot_reports_all_violations():
    """PITSnapshot reports all violations in a single error, not just the first."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    try:
        PITSnapshot(
            prices=pd.DataFrame(
                {"ANZ.NZ": [50.0, 51.0, 52.0]},
                index=pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"]),
            ),
            volumes=pd.DataFrame(
                {"ANZ.NZ": [100_000.0, 120_000.0, 130_000.0]},
                index=pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"]),
            ),
            fundamentals=pd.DataFrame(),
            macro=pd.DataFrame(),
            corporate_actions=pd.DataFrame(),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        msg = str(e).lower()
        assert "prices" in msg and "volumes" in msg, (
            f"Expected both 'prices' and 'volumes' in error, got: {e}"
        )


# ---------------------------------------------------------------------------
# PreparedPanel alignment invariants
# ---------------------------------------------------------------------------

def _make_panel(**overrides):
    """Build a minimal valid PreparedPanel; override any field via kwargs."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    rb_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-31")])
    tickers = ["ANZ.NZ", "SPK.NZ"]

    base = dict(
        returns_daily=pd.DataFrame(0.0, index=dates, columns=tickers),
        returns_monthly=pd.DataFrame(0.0, index=rb_dates, columns=tickers),
        market_cap=pd.DataFrame(1e9, index=dates, columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame(True, index=rb_dates, columns=tickers),
        macro=pd.DataFrame(),
        asof=pd.Timestamp("2025-01-01", tz="UTC"),
    )
    base.update(overrides)
    return PreparedPanel(**base)


def test_prepared_panel_valid_construction():
    panel = _make_panel()
    assert list(panel.sector.index) == ["ANZ.NZ", "SPK.NZ"]


def test_skuld_research_top_level_exports_are_lazy():
    """Importing `skuld_research` should not eagerly import deep submodules."""
    for name in [
        "skuld_research",
        "skuld_research.factors",
        "skuld_research.factors.momentum",
        "skuld_research.portfolio",
        "skuld_research.portfolio.optimizer",
    ]:
        sys.modules.pop(name, None)

    package = importlib.import_module("skuld_research")

    assert "skuld_research.factors" not in sys.modules
    assert "skuld_research.portfolio.optimizer" not in sys.modules

    assert package.MomentumFactor.__name__ == "MomentumFactor"
    assert package.build_target_portfolio.__name__ == "build_target_portfolio"

    assert "skuld_research.factors.momentum" in sys.modules
    assert "skuld_research.portfolio.optimizer" in sys.modules


@pytest.mark.parametrize(
    ("package_name", "root_prefix", "eager_modules", "attr_name", "loaded_module"),
    [
        (
            "skuld_common",
            "skuld_common",
            ["skuld_common.contracts", "skuld_common.validation"],
            "PITSnapshot",
            "skuld_common.contracts",
        ),
        (
            "skuld_research.data",
            "skuld_research",
            [
                "skuld_research.data.csv_loader",
                "skuld_research.data.pit_loader",
                "skuld_research.data.prepared_panel",
            ],
            "load_raw_csv",
            "skuld_research.data.csv_loader",
        ),
        (
            "skuld_research.config",
            "skuld_research",
            [
                "skuld_research.config.spec",
                "skuld_research.config.hashing",
                "skuld_research.config.loader",
                "skuld_research.config.runner",
            ],
            "load_spec",
            "skuld_research.config.loader",
        ),
        (
            "skuld_research.stats",
            "skuld_research",
            [
                "skuld_research.stats.bootstrap",
                "skuld_research.stats.deflated",
                "skuld_research.stats.dominance",
                "skuld_research.stats.gating",
                "skuld_research.stats.ledger",
                "skuld_research.stats.regimes",
                "skuld_research.stats.rolling_walk_forward",
            ],
            "evaluate",
            "skuld_research.stats.gating",
        ),
        (
            "skuld_research.backtest",
            "skuld_research",
            [
                "skuld_research.backtest.engine",
                "skuld_research.backtest.metrics",
                "skuld_research.backtest.walk_forward",
            ],
            "BacktestEngine",
            "skuld_research.backtest.engine",
        ),
        (
            "skuld_research.benchmarks",
            "skuld_research",
            [
                "skuld_research.benchmarks.nz_td_floor",
                "skuld_research.benchmarks.nzx_equal_weighted_fixed_universe",
                "skuld_research.benchmarks.sixty_forty",
            ],
            "nz_td_floor",
            "skuld_research.benchmarks.nz_td_floor",
        ),
        (
            "skuld_research.costs",
            "skuld_research",
            [
                "skuld_research.costs.model",
                "skuld_research.costs.spread_estimator",
            ],
            "CostModel",
            "skuld_research.costs.model",
        ),
        (
            "skuld_research.diagnostics",
            "skuld_research",
            [
                "skuld_research.diagnostics.decay",
                "skuld_research.diagnostics.decomposition",
                "skuld_research.diagnostics.ic",
                "skuld_research.diagnostics.report",
            ],
            "ranking_ic",
            "skuld_research.diagnostics.ic",
        ),
        (
            "skuld_research.factors",
            "skuld_research",
            [
                "skuld_research.factors.combiner",
                "skuld_research.factors.low_volatility",
                "skuld_research.factors.momentum",
                "skuld_research.factors.protocols",
                "skuld_research.factors.size",
            ],
            "combine_signals",
            "skuld_research.factors.combiner",
        ),
        (
            "skuld_research.overlay",
            "skuld_research",
            [
                "skuld_research.overlay.apply",
                "skuld_research.overlay.rules",
            ],
            "apply_cash_overlay",
            "skuld_research.overlay.apply",
        ),
        (
            "skuld_research.portfolio",
            "skuld_research",
            ["skuld_research.portfolio.optimizer"],
            "build_target_portfolio",
            "skuld_research.portfolio.optimizer",
        ),
        (
            "skuld_research.reporting",
            "skuld_research",
            [
                "skuld_research.reporting.report_builder",
                "skuld_research.reporting.markdown_writer",
            ],
            "build_methodology_report",
            "skuld_research.reporting.report_builder",
        ),
        (
            "skuld_research.survivorship",
            "skuld_research",
            [
                "skuld_research.backtest.metrics",
                "skuld_research.survivorship.bias",
            ],
            "SurvivorshipAdjuster",
            "skuld_research.survivorship.bias",
        ),
    ],
)
def test_package_re_exports_are_lazy(
    package_name: str,
    root_prefix: str,
    eager_modules: list[str],
    attr_name: str,
    loaded_module: str,
):
    """Package re-exports should not import their implementation modules eagerly."""
    for name in [module for module in list(sys.modules) if module.startswith(root_prefix)]:
        sys.modules.pop(name, None)

    package = importlib.import_module(package_name)

    for module_name in eager_modules:
        assert module_name not in sys.modules

    getattr(package, attr_name)

    assert loaded_module in sys.modules


def test_prepared_panel_rejects_mismatched_returns_monthly_columns():
    dates = pd.bdate_range("2024-01-02", periods=10)
    rb_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-31")])
    with pytest.raises(ValueError, match="returns_monthly"):
        _make_panel(
            returns_monthly=pd.DataFrame(
                0.0, index=rb_dates, columns=["ANZ.NZ", "WRONG.NZ"]
            )
        )


def test_prepared_panel_rejects_mismatched_market_cap_columns():
    dates = pd.bdate_range("2024-01-02", periods=10)
    with pytest.raises(ValueError, match="market_cap"):
        _make_panel(
            market_cap=pd.DataFrame(1e9, index=dates, columns=["ANZ.NZ"])
        )


def test_prepared_panel_rejects_mismatched_sector_index():
    with pytest.raises(ValueError, match="sector"):
        _make_panel(sector=pd.Series("Unknown", index=["ANZ.NZ"]))


def test_prepared_panel_rejects_universe_mask_with_unknown_tickers():
    rb_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-31")])
    with pytest.raises(ValueError, match="universe_mask"):
        _make_panel(
            universe_mask=pd.DataFrame(
                True, index=rb_dates, columns=["ANZ.NZ", "SPK.NZ", "UNKNOWN.NZ"]
            )
        )


# ---------------------------------------------------------------------------
# BacktestResult, FoldResult, WalkForwardResult
# ---------------------------------------------------------------------------

def test_backtest_result_construction():
    """BacktestResult constructs successfully with valid data."""
    dates = pd.date_range("2020-01-31", periods=12, freq="BME")
    result = BacktestResult(
        returns=pd.Series([0.01] * 12, index=dates),
        costs_nzd=pd.Series([50.0] * 12, index=dates),
        turnover=pd.Series([0.1] * 12, index=dates),
        drawdown=pd.Series([0.0] * 12, index=dates),
        sharpe_raw=1.5,
        sharpe_flat_haircut=1.1,
        start=dates[0],
        end=dates[-1],
        n_periods=12,
        avg_positions=10.0,
    )
    assert result.n_periods == 12
    assert result.avg_positions == 10.0


def test_backtest_result_rejects_negative_n_periods():
    dates = pd.date_range("2020-01-31", periods=6, freq="BME")
    with pytest.raises(ValueError, match="n_periods"):
        BacktestResult(
            returns=pd.Series([0.0] * 6, index=dates),
            costs_nzd=pd.Series([0.0] * 6, index=dates),
            turnover=pd.Series([0.0] * 6, index=dates),
            drawdown=pd.Series([0.0] * 6, index=dates),
            sharpe_raw=0.0,
            sharpe_flat_haircut=0.0,
            start=dates[0],
            end=dates[-1],
            n_periods=-1,
            avg_positions=0.0,
        )


def test_fold_result_rejects_invalid_date_order():
    dates = pd.date_range("2020-01-31", periods=6, freq="BME")
    br = BacktestResult(
        returns=pd.Series([0.0] * 6, index=dates),
        costs_nzd=pd.Series([0.0] * 6, index=dates),
        turnover=pd.Series([0.0] * 6, index=dates),
        drawdown=pd.Series([0.0] * 6, index=dates),
        sharpe_raw=0.0,
        sharpe_flat_haircut=0.0,
        start=dates[0],
        end=dates[-1],
        n_periods=6,
        avg_positions=0.0,
    )
    with pytest.raises(ValueError, match="test_start"):
        FoldResult(
            fold_id=0,
            test_start=dates[-1],   # AFTER test_end -> invalid
            test_end=dates[0],
            result=br,
        )


def test_walk_forward_result_valid_construction():
    dates = pd.date_range("2020-01-31", periods=12, freq="BME")
    br = BacktestResult(
        returns=pd.Series([0.005] * 12, index=dates),
        costs_nzd=pd.Series([20.0] * 12, index=dates),
        turnover=pd.Series([0.05] * 12, index=dates),
        drawdown=pd.Series([0.0] * 12, index=dates),
        sharpe_raw=1.0,
        sharpe_flat_haircut=0.7,
        start=dates[0],
        end=dates[-1],
        n_periods=12,
        avg_positions=8.0,
    )
    fold = FoldResult(fold_id=0, test_start=dates[0], test_end=dates[-1], result=br)
    wfr = WalkForwardResult(
        folds=(fold,),
        oos_returns=br.returns,
        oos_sharpe_raw=1.0,
        oos_sharpe_flat_haircut=0.7,
        oos_sharpe_delisting_adjusted=0.5,
        oos_drawdown_observed=br.drawdown,
        oos_max_drawdown_observed=0.0,   # flat -> 0 is OK
        oos_max_drawdown_augmented_median=-0.05,
        oos_max_drawdown_augmented_p90=-0.12,
        oos_avg_turnover=0.05,
        oos_total_cost_nzd=240.0,
    )
    assert len(wfr.folds) == 1


# ---------------------------------------------------------------------------
# CurrentPortfolio
# ---------------------------------------------------------------------------

def test_current_portfolio_construction():
    """CurrentPortfolio constructs successfully with aligned holdings and prices."""
    holdings = pd.Series([100, 200], index=["AIR", "FBU"], dtype=int)
    prices = pd.Series([2.50, 4.00], index=["AIR", "FBU"], dtype=float)
    cp = CurrentPortfolio(holdings=holdings, prices=prices, cash_nzd=1000.0)
    assert cp.cash_nzd == 1000.0
    assert len(cp.holdings) == 2


def test_current_portfolio_rejects_negative_shares():
    """CurrentPortfolio raises if holdings contain negative shares."""
    holdings = pd.Series([100, -50], index=["AIR", "FBU"], dtype=int)
    prices = pd.Series([2.50, 4.00], index=["AIR", "FBU"], dtype=float)
    with pytest.raises(ValueError, match="negative shares"):
        CurrentPortfolio(holdings=holdings, prices=prices, cash_nzd=1000.0)


def test_current_portfolio_rejects_negative_cash():
    """CurrentPortfolio raises if cash_nzd is negative."""
    holdings = pd.Series([100], index=["AIR"], dtype=int)
    prices = pd.Series([2.50], index=["AIR"], dtype=float)
    with pytest.raises(ValueError, match="cash_nzd"):
        CurrentPortfolio(holdings=holdings, prices=prices, cash_nzd=-100.0)


def test_current_portfolio_rejects_misaligned_indices():
    """CurrentPortfolio raises if holdings and prices indices don't match."""
    holdings = pd.Series([100, 200], index=["AIR", "FBU"], dtype=int)
    prices = pd.Series([2.50], index=["AIR"], dtype=float)  # Missing FBU
    with pytest.raises(ValueError, match="prices has tickers without holdings|holdings has tickers without prices"):
        CurrentPortfolio(holdings=holdings, prices=prices, cash_nzd=1000.0)


# ---------------------------------------------------------------------------
# TradeList
# ---------------------------------------------------------------------------

def test_trade_list_construction():
    """TradeList constructs successfully with all required columns."""
    trades_df = pd.DataFrame({
        "ticker": ["AIR", "FBU"],
        "action": ["BUY", "HOLD"],
        "current_shares": [0, 100],
        "target_shares": [50, 100],
        "delta_shares": [50, 0],
        "current_value_nzd": [0.0, 400.0],
        "target_value_nzd": [125.0, 400.0],
        "delta_value_nzd": [125.0, 0.0],
        "est_round_trip_cost_nzd": [2.50, 0.0],
        "in_no_trade_region": [False, True],
        "below_size_floor": [False, False],
        "deferred_to_next_month": [False, False],
        "sharesies_fee_band": ["flat_15", "subscription_only"],
    })
    tl = TradeList(
        trades=trades_df,
        total_volume_nzd=125.0,
        total_estimated_cost_nzd=17.50,
        asof=pd.Timestamp("2026-01-01", tz="UTC"),
        config_hash="abc123",
    )
    assert tl.total_volume_nzd == 125.0
    assert len(tl.trades) == 2


def test_trade_list_rejects_missing_columns():
    """TradeList raises if required columns are missing."""
    trades_df = pd.DataFrame({
        "ticker": ["AIR"],
        "action": ["BUY"],
        # Missing all other required columns
    })
    with pytest.raises(ValueError, match="missing required columns"):
        TradeList(
            trades=trades_df,
            total_volume_nzd=100.0,
            total_estimated_cost_nzd=15.0,
            asof=pd.Timestamp("2026-01-01", tz="UTC"),
            config_hash="abc",
        )
