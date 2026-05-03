"""
Walk-forward evaluation: rolling OOS Sharpe, fold table, regime breakdown.

Usage (from repo root):
    uv run --project python python scripts/walk_forward_eval.py [path/to/spec.yaml]

Default spec: python/configs/strategy-specs/candidates/mom-s7.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python" / "src"))

from skuld_research.config.loader import load_spec
from skuld_research.config.factors import build_factors_from_specs
from skuld_research.data.csv_loader import load_raw_csv, load_raw_ohlc
from skuld_research.data.pit_loader import PITLoader
from skuld_research.data.prepared_panel import build_prepared_panel
from skuld_research.backtest.engine import BacktestConfig, BacktestEngine
from skuld_research.costs.model import CostConfig
from skuld_research.costs.spread_estimator import compute_abdi_ranaldo_spread_panel
from skuld_research.stats.rolling_walk_forward import RollingWalkForwardEngine

DATA_PATH = _REPO_ROOT / "data" / "data_long.csv"
_DEFAULT_SPEC = _REPO_ROOT / "python" / "configs" / "strategy-specs" / "candidates" / "mom-s7.yaml"
_PYTHON_ROOT = _REPO_ROOT / "python"


def _label_spread_by_next_observation(spread_panel: pd.DataFrame) -> pd.DataFrame:
    """Relabel each spread row to the next observed source row (PIT-safe)."""
    if spread_panel.empty:
        return spread_panel
    relabelled = spread_panel.iloc[:-1].copy()
    relabelled.index = spread_panel.index[1:]
    return relabelled


def _build_spread_panel(spec, raw_csv_path: Path) -> pd.DataFrame | None:
    if spec.cost.spread_model != "abdi_ranaldo":
        return None
    print("  Building Abdi-Ranaldo spread panel from OHLC...")
    high, low, close = load_raw_ohlc(raw_csv_path, scrub=spec.scrubbing, adjustments=spec.adjustments)
    sp = compute_abdi_ranaldo_spread_panel(
        high, low, close,
        window=spec.cost.spread_estimator_window,
        min_obs=spec.cost.spread_estimator_min_obs,
        scale=spec.cost.spread_estimator_scale,
        min_bps_per_side=spec.cost.spread_estimator_min_bps_per_side,
    )
    if isinstance(sp.index, pd.DatetimeIndex):
        sp = _label_spread_by_next_observation(sp)
    return sp


def _build_backtest_config(spec, cost_config: CostConfig) -> BacktestConfig:
    return BacktestConfig(
        initial_nav_nzd=spec.backtest.initial_nav_nzd,
        cash_floor=spec.backtest.cash_floor,
        max_position=spec.backtest.max_position,
        max_sector=spec.backtest.max_sector,
        min_names=spec.backtest.min_names,
        score_lambda=spec.backtest.score_lambda,
        no_trade_threshold_frac=spec.backtest.no_trade_threshold_frac,
        size_floor_nzd=spec.backtest.size_floor_nzd,
        size_floor_cost_multiple=spec.backtest.size_floor_cost_multiple,
        return_window_days=spec.backtest.return_window_days,
        min_return_obs=spec.backtest.min_return_obs,
        cost_config=cost_config,
        flat_haircut_bps=spec.backtest.flat_haircut_bps,
        risk_free_annual=spec.backtest.risk_free_annual,
        min_positions_per_month=spec.backtest.min_positions_per_month,
        degenerate_fold_max_empty_frac=spec.backtest.degenerate_fold_max_empty_frac,
        turnover_budget_frac=spec.backtest.turnover_budget_frac,
        smoothing_alpha=getattr(spec.backtest, "smoothing_alpha", 0.0),
    )


def _bootstrap_sharpe_ci(returns: pd.Series, n_boot: int = 2000, ci: float = 0.95, rf: float = 0.0) -> tuple[float, float]:
    """Percentile bootstrap CI for annualised Sharpe."""
    rng = np.random.default_rng(42)
    n = len(returns)
    arr = returns.values
    boot_sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        mu = sample.mean() * 12.0
        vol = sample.std(ddof=1) * (12.0 ** 0.5)
        s = (mu - rf) / vol if vol > 1e-12 else 0.0
        boot_sharpes.append(s)
    lo = np.percentile(boot_sharpes, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_sharpes, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def print_fold_table(wf_result) -> None:
    """Print per-fold results table."""
    print(f"\n{'Fold':>4}  {'Test Start':>10}  {'Test End':>10}  {'Sharpe':>7}  {'Hit%':>6}  {'AvgPos':>7}")
    print("-" * 60)
    for fr in wf_result.folds:
        r = fr.result
        sharpe = r.sharpe_raw
        hit = r.hit_rate
        avg_pos = r.avg_positions
        print(
            f"{fr.fold_id:>4}  "
            f"{fr.test_start:%Y-%m-%d}  "
            f"{fr.test_end:%Y-%m-%d}  "
            f"{sharpe:>7.3f}  "
            f"{hit:>5.1%}  "
            f"{avg_pos:>7.1f}"
        )


def print_aggregate_metrics(wf_result, rf: float = 0.0) -> None:
    """Print aggregate OOS metrics."""
    r = wf_result
    print(f"\n{'=' * 52}")
    print("  Aggregate OOS Metrics")
    print(f"{'=' * 52}")
    print(f"  OOS Sharpe (raw):               {r.oos_sharpe_raw:>8.3f}")
    print(f"  OOS Sharpe (flat haircut):       {r.oos_sharpe_flat_haircut:>8.3f}")
    print(f"  OOS Sharpe (delisting adj):      {r.oos_sharpe_delisting_adjusted:>8.3f}")
    print(f"  OOS Hit rate:                    {r.oos_hit_rate:>8.1%}")
    print(f"  OOS Calmar ratio:                {r.oos_calmar_ratio:>8.3f}")
    print(f"  OOS Max drawdown (observed):     {r.oos_max_drawdown_observed:>8.1%}")
    print(f"  OOS Max drawdown (MC median):    {r.oos_max_drawdown_augmented_median:>8.1%}")
    print(f"  OOS Avg turnover:                {r.oos_avg_turnover:>8.1%}")
    print(f"  N kept folds:                    {r.n_kept_folds:>8d}")
    print(f"  N rejected folds:                {r.n_rejected_folds:>8d}")

    if r.rejection_reasons:
        for reason in r.rejection_reasons:
            print(f"    Rejected: {reason}")

    # Bootstrap CI
    if not r.oos_returns.empty and len(r.oos_returns) >= 4:
        lo, hi = _bootstrap_sharpe_ci(r.oos_returns, rf=rf)
        print(f"\n  Bootstrap 95% CI for OOS Sharpe: [{lo:.3f}, {hi:.3f}]")

    # Per-regime Sharpes
    if r.oos_sharpe_by_regime:
        print(f"\n  Regime Sharpes:")
        for regime, sharpe in r.oos_sharpe_by_regime.items():
            print(f"    {regime:<8}: {sharpe:.3f}")


def print_comparison(is_sharpe: float, oos_sharpe: float, spec_name: str) -> None:
    """Print in-sample vs OOS comparison."""
    print(f"\n{'=' * 52}")
    print("  Overfitting Check (IS vs OOS)")
    print(f"{'=' * 52}")
    print(f"  Full-sample IS Sharpe ({spec_name}): {is_sharpe:.3f}")
    print(f"  Rolling OOS Sharpe:                  {oos_sharpe:.3f}")
    ratio = oos_sharpe / is_sharpe if abs(is_sharpe) > 1e-9 else float("nan")
    print(f"  OOS/IS ratio:                        {ratio:.2f}")
    if ratio < 0.5:
        print("  [WARNING] OOS Sharpe < 50% of IS — significant overfitting suspected")
    elif ratio < 0.8:
        print("  [CAUTION] OOS Sharpe 50-80% of IS — moderate degradation")
    else:
        print("  [OK] OOS Sharpe >= 80% of IS — robust OOS performance")


def main() -> None:
    SPEC_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_SPEC
    print(f"=== Walk-Forward Evaluation: {SPEC_PATH.name} ===\n")

    print("Loading spec and raw data...")
    spec = load_spec(SPEC_PATH)
    raw = load_raw_csv(DATA_PATH, scrub=spec.scrubbing, adjustments=spec.adjustments)

    print(f"Building PIT snapshot (asof={spec.asof})...")
    snap = PITLoader(raw).as_of(pd.Timestamp(spec.asof, tz="UTC"))

    print(f"Building prepared panel...")
    panel = build_prepared_panel(
        snap,
        min_adv_dollars=spec.universe.min_adv_dollars,
        min_market_cap_nzd=spec.universe.min_market_cap_nzd,
        min_history_days=spec.universe.min_history_days,
        adv_window=spec.universe.adv_window,
        mc_ffill_days=spec.universe.mc_ffill_days,
        nzx_only=spec.universe.nzx_only,
        anomaly_filter=spec.anomaly_filter,
    )

    n_dates = len(panel.universe_mask.index)
    date_start = panel.universe_mask.index[0]
    date_end = panel.universe_mask.index[-1]
    print(f"  Panel: {date_start:%Y-%m} to {date_end:%Y-%m} ({n_dates} rebalance dates)")

    # Build spread panel if needed
    spread_panel = _build_spread_panel(spec, DATA_PATH)

    # Build cost + backtest configs
    cost_config = CostConfig(
        spread_bps=spec.cost.spread_bps,
        sharesies_monthly_fee_nzd=spec.cost.sharesies_monthly_fee_nzd,
        sharesies_coverage_nzd=spec.cost.sharesies_coverage_nzd,
        sharesies_excess_bps=spec.cost.sharesies_excess_bps,
    )
    backtest_config = _build_backtest_config(spec, cost_config)

    # Build factors
    factors = build_factors_from_specs(spec.factors)

    # Resolve delisting CSV
    delisting_csv_path = _PYTHON_ROOT / spec.survivorship.delisting_csv_relpath

    # --- Run rolling walk-forward ---
    print(f"\nRunning rolling walk-forward "
          f"(train={spec.walk_forward.rolling.train_years}y / "
          f"oos={spec.walk_forward.rolling.oos_years}y / "
          f"step={spec.walk_forward.rolling.step_years}y)...")

    wf_engine = RollingWalkForwardEngine(
        panel=panel,
        factors=factors,
        train_years=spec.walk_forward.rolling.train_years,
        oos_years=spec.walk_forward.rolling.oos_years,
        step_years=spec.walk_forward.rolling.step_years,
        delisting_csv_path=delisting_csv_path,
        backtest_config=backtest_config,
        monte_carlo_seeds=spec.survivorship.monte_carlo_seeds,
        spread_panel=spread_panel,
    )
    wf_result = wf_engine.run()
    print(f"  Done — {len(wf_result.folds)} folds evaluated.")

    # --- Print fold table ---
    print_fold_table(wf_result)

    # --- Print aggregate metrics ---
    print_aggregate_metrics(wf_result, rf=spec.backtest.risk_free_annual)

    # --- Full-sample IS Sharpe for comparison ---
    print("\nRunning full-sample backtest for IS comparison...")
    is_engine = BacktestEngine(factors=factors, panel=panel, config=backtest_config, spread_panel=spread_panel)
    is_result = is_engine.run()

    print_comparison(is_result.sharpe_raw, wf_result.oos_sharpe_raw, SPEC_PATH.stem)

    print()


if __name__ == "__main__":
    main()
