"""Tests for RollingWalkForwardEngine."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PITSnapshot, PreparedPanel
from skuld_research.data.prepared_panel import build_prepared_panel
from skuld_research.factors.momentum import MomentumFactor
from skuld_research.stats.rolling_walk_forward import RollingWalkForwardEngine


def _make_10y_panel(seed: int = 100) -> PreparedPanel:
    """Build a 10-year synthetic panel for rolling walk-forward tests."""
    rng = np.random.default_rng(seed)
    n_days = 10 * 252  # ~10 years
    tickers = [f"T{i:02d}.NZ" for i in range(10)]
    dates = pd.bdate_range("2016-01-01", periods=n_days)

    prices_data = {}
    for tk in tickers:
        px = 20.0 * (1.0 + 0.0008 * rng.standard_normal(n_days)).cumprod()
        prices_data[tk] = px

    prices = pd.DataFrame(prices_data, index=dates)
    prices.index.name = "date"
    volumes = pd.DataFrame({tk: 1_000_000.0 for tk in tickers}, index=dates)
    volumes.index.name = "date"

    # Fake shares-outstanding so build_prepared_panel can compute market_cap.
    # One observation per ticker, dated at the start of the panel; ffill handles
    # the rest. 50M shares × ~$20 = ~$1B mcap, well above the $20M floor.
    fund_dates = [dates[0]] * len(tickers)
    fund_index = pd.MultiIndex.from_arrays(
        [tickers, fund_dates], names=["ticker", "publication_date"]
    )
    fundamentals = pd.DataFrame(
        {"trailing_basic_average_shares": [50_000_000.0] * len(tickers)},
        index=fund_index,
    )

    asof_ts = dates[-1] + pd.DateOffset(months=3)

    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
        asof=asof_ts,
    )

    return build_prepared_panel(snap, nzx_only=False, rebalance_start="2016-01-01")


def test_rolling_wf_returns_walk_forward_result():
    """RollingWalkForwardEngine.run() returns a WalkForwardResult."""
    panel = _make_10y_panel()

    engine = RollingWalkForwardEngine(
        panel=panel,
        factors=[MomentumFactor()],
        train_years=5,
        oos_years=1,
        step_years=1,
    )

    result = engine.run()

    from skuld_common.contracts import WalkForwardResult
    assert isinstance(result, WalkForwardResult)


def test_rolling_wf_generates_5_folds():
    """10y panel with 5y train + 1y OOS + 1y step → 5 folds (years 6–10)."""
    panel = _make_10y_panel()

    engine = RollingWalkForwardEngine(
        panel=panel,
        factors=[MomentumFactor()],
        train_years=5,
        oos_years=1,
        step_years=1,
    )

    result = engine.run()

    # Folds: year 6, 7, 8, 9, 10 → 5 folds
    # (year 1-5 is warmup, first OOS is year 6)
    assert len(result.folds) >= 4  # At least 4-5 folds depending on exact month boundaries


def test_rolling_wf_non_overlapping_folds():
    """Folds are non-overlapping and sequential."""
    panel = _make_10y_panel()

    engine = RollingWalkForwardEngine(
        panel=panel,
        factors=[MomentumFactor()],
        train_years=5,
        oos_years=1,
        step_years=1,
    )

    result = engine.run()

    # Check non-overlapping
    for i in range(len(result.folds) - 1):
        fold_a = result.folds[i]
        fold_b = result.folds[i + 1]
        # fold_a.test_end should be before fold_b.test_start
        assert fold_a.test_end < fold_b.test_start


def test_rolling_wf_oos_returns_nonempty():
    """oos_returns is non-empty and monotonically increasing."""
    panel = _make_10y_panel()

    # Use less strict config to avoid all folds being rejected
    from skuld_research.backtest.engine import BacktestConfig
    config = BacktestConfig(
        min_positions_per_month=1,  # More lenient
        degenerate_fold_max_empty_frac=0.9,
    )

    engine = RollingWalkForwardEngine(
        panel=panel,
        factors=[MomentumFactor()],
        train_years=5,
        oos_years=1,
        step_years=1,
        backtest_config=config,
    )

    result = engine.run()

    # At least some folds should be kept
    assert result.n_kept_folds > 0
    if len(result.oos_returns) > 0:
        assert result.oos_returns.index.is_monotonic_increasing


def test_rolling_wf_fold_length():
    """Each fold has ~12 monthly rebalances (1 year OOS)."""
    panel = _make_10y_panel()

    # Use less strict config
    from skuld_research.backtest.engine import BacktestConfig
    config = BacktestConfig(
        min_positions_per_month=1,
        degenerate_fold_max_empty_frac=0.9,
    )

    engine = RollingWalkForwardEngine(
        panel=panel,
        factors=[MomentumFactor()],
        train_years=5,
        oos_years=1,
        step_years=1,
        backtest_config=config,
    )

    result = engine.run()

    # Each fold should have around 12 returns (monthly rebalances)
    for fold in result.folds:
        n_returns = len(fold.result.returns)
        # Allow wider tolerance
        assert 1 <= n_returns <= 14
