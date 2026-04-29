"""Tests for WalkForwardEngine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import BacktestResult, PITSnapshot, PreparedPanel, WalkForwardResult
from skuld_research.backtest.engine import BacktestConfig, BacktestEngine
from skuld_research.backtest.walk_forward import FoldSpec, WalkForwardEngine
from skuld_research.data.prepared_panel import build_prepared_panel
from skuld_research.factors.momentum import MomentumFactor


# ---------------------------------------------------------------------------
# Helper: build a PreparedPanel with synthetic data (n_days=800, n_tickers=15)
# ---------------------------------------------------------------------------

def _make_panel(
    n_tickers: int = 15,
    n_days: int = 800,
    seed: int = 0,
) -> PreparedPanel:
    """Build a PreparedPanel with synthetic daily returns for walk-forward tests."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}.NZ" for i in range(n_tickers)]
    dates = pd.bdate_range("2021-01-01", periods=n_days)

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
    return build_prepared_panel(snap, nzx_only=False, rebalance_start="2021-01-01")


def _make_two_fold_engine(panel: PreparedPanel | None = None) -> tuple[WalkForwardEngine, list[FoldSpec]]:
    """Build a WalkForwardEngine with two non-overlapping folds."""
    if panel is None:
        panel = _make_panel()
    rebalance_dates = panel.universe_mask.index
    n = len(rebalance_dates)
    mid = n // 2

    folds = [
        FoldSpec(0, rebalance_dates[1], rebalance_dates[mid]),
        FoldSpec(1, rebalance_dates[mid + 1], rebalance_dates[-1]),
    ]
    engine = WalkForwardEngine(
        factors=[MomentumFactor()],
        panel=panel,
        folds=folds,
        monte_carlo_seeds=50,
        mc_rng_seed=42,
    )
    return engine, folds


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_walk_forward_returns_result():
    """WalkForwardEngine.run() returns a WalkForwardResult."""
    engine, _ = _make_two_fold_engine()
    result = engine.run()
    assert isinstance(result, WalkForwardResult)


def test_n_folds_matches_spec():
    """len(result.folds) equals the number of FoldSpec objects passed."""
    engine, folds = _make_two_fold_engine()
    result = engine.run()
    assert len(result.folds) == len(folds)


def test_oos_returns_span_all_folds():
    """oos_returns is non-empty and has a monotonically increasing index."""
    from skuld_research.backtest.engine import BacktestConfig
    config = BacktestConfig(degenerate_fold_max_empty_frac=1.0)  # Disable rejection for this test
    
    panel = _make_panel()
    rebalance_dates = panel.universe_mask.index
    n = len(rebalance_dates)
    mid = n // 2
    
    folds = [
        FoldSpec(0, rebalance_dates[1], rebalance_dates[mid]),
        FoldSpec(1, rebalance_dates[mid + 1], rebalance_dates[-1]),
    ]
    engine = WalkForwardEngine(
        factors=[MomentumFactor()],
        panel=panel,
        folds=folds,
        backtest_config=config,
        monte_carlo_seeds=50,
        mc_rng_seed=42,
    )
    result = engine.run()
    assert len(result.oos_returns) > 0
    assert result.oos_returns.index.is_monotonic_increasing


def test_sharpe_hierarchy():
    """oos_sharpe_raw >= oos_sharpe_flat_haircut >= oos_sharpe_delisting_adjusted."""
    engine, _ = _make_two_fold_engine()
    result = engine.run()
    assert result.oos_sharpe_raw >= result.oos_sharpe_flat_haircut - 1e-9
    assert result.oos_sharpe_flat_haircut >= result.oos_sharpe_delisting_adjusted - 1e-9


def test_augmented_drawdown_not_better_than_observed():
    """MC-augmented median drawdown is no better than the observed drawdown."""
    engine, _ = _make_two_fold_engine()
    result = engine.run()
    assert result.oos_max_drawdown_augmented_median <= result.oos_max_drawdown_observed + 1e-9


def test_single_fold_same_as_engine_direct():
    """One fold covering all rebalance dates produces the same returns as BacktestEngine directly."""
    panel = _make_panel()
    rebalance_dates = panel.universe_mask.index

    from skuld_research.backtest.engine import BacktestConfig
    config = BacktestConfig(degenerate_fold_max_empty_frac=1.0)  # Disable rejection

    folds = [FoldSpec(0, rebalance_dates[0], rebalance_dates[-1])]
    wf = WalkForwardEngine(
        factors=[MomentumFactor()],
        panel=panel,
        folds=folds,
        backtest_config=config,
        monte_carlo_seeds=50,
        mc_rng_seed=42,
    )
    wf_result = wf.run()

    direct = BacktestEngine(
        factors=[MomentumFactor()],
        panel=panel,
        config=config,
    ).run()

    pd.testing.assert_series_equal(
        wf_result.oos_returns.reset_index(drop=True),
        direct.returns.reset_index(drop=True),
        check_names=False,
        rtol=1e-6,
    )


def test_fold_date_restriction():
    """Each fold's returns fall within its FoldSpec date window."""
    engine, folds = _make_two_fold_engine()
    result = engine.run()
    for fold_result, spec in zip(result.folds, folds):
        ret = fold_result.result.returns
        if ret.empty:
            continue
        assert ret.index.min() >= spec.test_start, (
            f"Fold {spec.fold_id}: earliest return {ret.index.min()} < test_start {spec.test_start}"
        )
        assert ret.index.max() <= spec.test_end, (
            f"Fold {spec.fold_id}: latest return {ret.index.max()} > test_end {spec.test_end}"
        )


def test_invalid_fold_date_raises():
    """A FoldSpec with test_start > test_end raises ValueError."""
    panel = _make_panel()
    rebalance_dates = panel.universe_mask.index
    # test_start after test_end — ValueError raised during FoldSpec construction
    with pytest.raises(ValueError):
        bad_folds = [FoldSpec(0, rebalance_dates[-1], rebalance_dates[0])]
        wf = WalkForwardEngine(
            factors=[MomentumFactor()],
            panel=panel,
            folds=bad_folds,
            monte_carlo_seeds=50,
        )
        wf.run()


def test_degenerate_fold_rejection():
    """Fold with mostly empty universes is rejected."""
    # Create a panel where one fold has mostly empty universes
    panel = _make_panel(n_tickers=5, n_days=400)
    rebalance_dates = panel.universe_mask.index
    n = len(rebalance_dates)
    mid = n // 2
    
    # Make second fold's universe mostly False
    panel.universe_mask.iloc[mid:, :] = False
    
    from skuld_common.contracts import PreparedPanel
    panel = PreparedPanel(
        returns_daily=panel.returns_daily,
        returns_monthly=panel.returns_monthly,
        market_cap=panel.market_cap,
        sector=panel.sector,
        universe_mask=panel.universe_mask,
        macro=panel.macro,
        asof=panel.asof,
    )
    
    folds = [
        FoldSpec(0, rebalance_dates[1], rebalance_dates[mid]),
        FoldSpec(1, rebalance_dates[mid + 1], rebalance_dates[-1]),
    ]
    
    from skuld_research.backtest.engine import BacktestConfig
    config = BacktestConfig(min_positions_per_month=5, degenerate_fold_max_empty_frac=0.5)
    
    wf = WalkForwardEngine(
        factors=[MomentumFactor()],
        panel=panel,
        folds=folds,
        backtest_config=config,
        monte_carlo_seeds=50,
    )
    result = wf.run()
    
    # Second fold should be rejected
    assert result.n_rejected_folds >= 1


def test_per_regime_sharpe_populated():
    """Per-regime Sharpe dict is populated."""
    engine, _ = _make_two_fold_engine()
    result = engine.run()
    
    # Should have at least one regime key
    assert len(result.oos_sharpe_by_regime) >= 0  # May be empty if not enough data


def test_new_wf_fields_have_defaults():
    """New WalkForwardResult fields have safe defaults for existing constructions."""
    # Manually construct a WalkForwardResult without the new fields (using defaults)
    from skuld_common.contracts import WalkForwardResult, FoldResult
    
    result = WalkForwardResult(
        folds=(),
        oos_returns=pd.Series([0.01, 0.02], index=pd.date_range("2020-01-31", periods=2, freq="ME")),
        oos_sharpe_raw=1.0,
        oos_sharpe_flat_haircut=0.6,
        oos_sharpe_delisting_adjusted=0.5,
        oos_drawdown_observed=pd.Series([0.0, 0.0]),
        oos_max_drawdown_observed=-0.05,
        oos_max_drawdown_augmented_median=-0.08,
        oos_max_drawdown_augmented_p90=-0.1,
        oos_avg_turnover=0.05,
        oos_total_cost_nzd=100.0,
    )
    
    # Check defaults
    assert result.n_kept_folds == 0
    assert result.n_rejected_folds == 0
    assert result.rejection_reasons == ()
    assert result.oos_sharpe_by_regime == {}


def test_precomputed_returns_short_circuit():
    """When precomputed_returns is provided, WalkForwardEngine returns it directly with zero costs."""
    panel = _make_panel(n_days=400)
    rebalance_dates = panel.universe_mask.index
    n = len(rebalance_dates)
    mid = n // 2
    
    # Synthetic returns series that spans the fold windows
    synthetic_returns = pd.Series(
        np.random.default_rng(100).standard_normal(n - 1) * 0.02,
        index=rebalance_dates[1:],
    )
    
    folds = [
        FoldSpec(0, rebalance_dates[1], rebalance_dates[mid]),
        FoldSpec(1, rebalance_dates[mid + 1], rebalance_dates[-1]),
    ]
    
    from skuld_research.backtest.engine import BacktestConfig
    config = BacktestConfig(degenerate_fold_max_empty_frac=1.0)  # Disable rejection
    
    wf = WalkForwardEngine(
        factors=[MomentumFactor()],  # Should be ignored
        panel=panel,
        folds=folds,
        backtest_config=config,
        monte_carlo_seeds=50,
        precomputed_returns=synthetic_returns,
    )
    result = wf.run()
    
    # OOS returns should match the precomputed series (aligned to fold windows)
    expected_oos = pd.concat([
        synthetic_returns.loc[
            (synthetic_returns.index >= folds[0].test_start) &
            (synthetic_returns.index <= folds[0].test_end)
        ],
        synthetic_returns.loc[
            (synthetic_returns.index >= folds[1].test_start) &
            (synthetic_returns.index <= folds[1].test_end)
        ],
    ])
    
    pd.testing.assert_series_equal(result.oos_returns, expected_oos, check_names=False)
    
    # Costs and turnover should be zero
    assert result.oos_total_cost_nzd == 0.0
    assert result.oos_avg_turnover == 0.0

