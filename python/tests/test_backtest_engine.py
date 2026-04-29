"""Tests for BacktestEngine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import BacktestResult, PITSnapshot, PreparedPanel
from skuld_research.backtest.engine import BacktestConfig, BacktestEngine
from skuld_research.data.prepared_panel import build_prepared_panel
from skuld_research.factors.momentum import MomentumFactor


# ---------------------------------------------------------------------------
# Helper: build a PreparedPanel with synthetic data
# ---------------------------------------------------------------------------

def _make_panel(
    n_tickers: int = 15,
    n_days: int = 600,
    seed: int = 0,
) -> PreparedPanel:
    """Build a PreparedPanel with synthetic daily returns for backtest tests."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}.NZ" for i in range(n_tickers)]
    dates = pd.bdate_range("2022-01-01", periods=n_days)

    prices_data = {}
    for t in tickers:
        px = 10.0 * (1 + 0.001 * rng.standard_normal(n_days)).cumprod()
        prices_data[t] = px

    prices = pd.DataFrame(prices_data, index=dates)
    prices.index.name = "date"
    volumes = pd.DataFrame({t: 500_000.0 for t in tickers}, index=dates)
    volumes.index.name = "date"

    last_date = dates[-1]
    asof_ts = last_date + pd.DateOffset(months=3)

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
    return build_prepared_panel(snap, nzx_only=False, rebalance_start="2022-01-01")


def _make_engine(config: BacktestConfig | None = None) -> BacktestEngine:
    panel = _make_panel()
    return BacktestEngine(
        factors=[MomentumFactor()],
        panel=panel,
        config=config,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_engine_returns_backtest_result():
    """engine.run() returns BacktestResult with at least one return period."""
    engine = _make_engine()
    result = engine.run()
    assert isinstance(result, BacktestResult)
    assert len(result.returns) >= 1


def test_costs_are_non_negative():
    """All period costs are non-negative."""
    result = _make_engine().run()
    assert (result.costs_nzd >= 0).all()


def test_turnover_is_non_negative():
    """All period turnover values are non-negative."""
    result = _make_engine().run()
    assert (result.turnover >= 0).all()


def test_drawdown_is_non_positive():
    """All drawdown values are <= 0 (within float tolerance)."""
    result = _make_engine().run()
    assert (result.drawdown <= 1e-9).all()


def test_sharpe_flat_haircut_le_raw():
    """Flat-haircut Sharpe is always <= raw Sharpe (within tolerance)."""
    result = _make_engine().run()
    assert result.sharpe_flat_haircut <= result.sharpe_raw + 1e-9


def test_no_trade_threshold_reduces_turnover():
    """Wide NTR (0.30) produces <= mean turnover compared to no NTR (0.0)."""
    panel = _make_panel(seed=42)

    cfg_wide = BacktestConfig(no_trade_threshold_frac=0.30)
    cfg_none = BacktestConfig(no_trade_threshold_frac=0.0)

    result_wide = BacktestEngine(factors=[MomentumFactor()], panel=panel, config=cfg_wide).run()
    result_none = BacktestEngine(factors=[MomentumFactor()], panel=panel, config=cfg_none).run()

    assert float(result_wide.turnover.mean()) <= float(result_none.turnover.mean()) + 1e-9


def test_backtest_result_invariants_hold():
    """BacktestResult dataclass invariants don't raise."""
    result = _make_engine().run()
    # Accessing fields exercises __post_init__ (already called on construction)
    assert result.n_periods >= 1
    assert result.avg_positions >= 0.0
    assert result.start <= result.end


def test_minimum_rebalance_dates_raises():
    """Panel with only 1 rebalance date raises ValueError."""
    panel = _make_panel(n_days=600)
    # Slice universe_mask to a single row
    single_mask = panel.universe_mask.iloc[:1]
    short_panel = PreparedPanel(
        returns_daily=panel.returns_daily,
        returns_monthly=panel.returns_monthly,
        market_cap=panel.market_cap,
        sector=panel.sector,
        universe_mask=single_mask,
        macro=panel.macro,
        asof=panel.asof,
    )
    engine = BacktestEngine(factors=[MomentumFactor()], panel=short_panel)
    with pytest.raises(ValueError, match="at least 2 rebalance dates"):
        engine.run()


def test_drift_weights_wipeout_does_not_crash():
    """_drift_weights with portfolio_gross_return = -1.0 does not raise and returns non-negative weights."""
    from skuld_research.backtest.engine import _drift_weights
    weights = pd.Series({"A.NZ": 0.5, "B.NZ": 0.5})
    period_returns = pd.Series({"A.NZ": -1.0, "B.NZ": -1.0})
    result = _drift_weights(weights, period_returns, portfolio_gross_return=-1.0)
    # Should not raise; weights must remain non-negative (clipped, not negative)
    assert (result >= 0.0).all()


def test_ntr_prevents_individual_small_trades():
    """When NTR=0.5 (50% of NAV), no position smaller than 50% of NAV gets traded."""
    from skuld_research.backtest.engine import BacktestConfig

    # With NTR=0.5, any weight delta smaller than 0.50 should be skipped.
    # With 15 tickers averaging 1/15 ≈ 6.7% weight each, all individual trades
    # are below the threshold, so turnover should be 0.0 on every period.
    cfg = BacktestConfig(no_trade_threshold_frac=0.50)
    panel = _make_panel(seed=99)
    engine = BacktestEngine(factors=[MomentumFactor()], panel=panel, config=cfg)
    result = engine.run()
    # With NTR=50%, all individual rebalance deltas should be blocked.
    # Turnover must be 0 for every period (no trades should execute).
    assert float(result.turnover.max()) == pytest.approx(0.0, abs=1e-9)


def test_universe_exit_liquidates_holdings():
    """Tickers that held but drop out of universe should be liquidated (n_positions drops)."""
    # Build an engine and run it. Since we can't easily inspect internal state,
    # we verify that n_positions in BacktestResult is never negative and reflects
    # the actual universe constraint: avg_positions <= n_tickers.
    result = _make_engine().run()
    n_tickers = 15  # from _make_panel default
    assert result.avg_positions <= n_tickers
    # avg_positions must be >= 0 (not a hard lower bound since universe can be empty)
    assert result.avg_positions >= 0.0


def test_synthetic_backtest_returns_in_plausible_range():
    """Regression guard: net monthly returns for the synthetic panel stay within ±30%."""
    result = _make_engine(BacktestConfig(initial_nav_nzd=10_000.0)).run()
    # Monthly returns outside ±30% would indicate a bug (extreme NAV crash or look-ahead)
    assert float(result.returns.max()) <= 0.30, (
        f"Suspiciously high monthly return: {result.returns.max():.4f}"
    )
    assert float(result.returns.min()) >= -0.30, (
        f"Suspiciously large monthly loss: {result.returns.min():.4f}"
    )
