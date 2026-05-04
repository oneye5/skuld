"""
Benchmark comparison: mom-s8 vs three benchmarks (OOS period).

Usage (from repo root):
    uv run python scripts/benchmark_comparison.py [path/to/spec.yaml]
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
from skuld_research.data.csv_loader import load_raw_csv
from skuld_research.data.pit_loader import PITLoader
from skuld_research.data.prepared_panel import build_prepared_panel
from skuld_research.backtest.engine import BacktestConfig, BacktestEngine
from skuld_research.costs.model import CostConfig
from skuld_research.costs.spread_estimator import compute_abdi_ranaldo_spread_panel
from skuld_research.stats.rolling_walk_forward import RollingWalkForwardEngine
from skuld_research.backtest.benchmark import (
    compute_sixty_forty_returns,
    compute_td_floor_returns,
    compute_benchmark_stats,
)
from skuld_research.data.csv_loader import load_raw_ohlc

DATA_PATH = _REPO_ROOT / "data" / "data_long.csv"
_DEFAULT_SPEC = _REPO_ROOT / "python" / "configs" / "strategy-specs" / "candidates" / "mom-s8.yaml"
_PYTHON_ROOT = _REPO_ROOT / "python"


def _annualised_sharpe(returns: pd.Series, rf_annual: float = 0.0) -> float:
    rf_monthly = rf_annual / 12.0
    excess = returns - rf_monthly
    if len(excess) < 2 or excess.std(ddof=1) < 1e-12:
        return float("nan")
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(12))


def _max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    return float(dd.min())


def _mean_annual(returns: pd.Series) -> float:
    return float(returns.mean() * 12)


def _label_spread_by_next_observation(spread_panel: pd.DataFrame) -> pd.DataFrame:
    if spread_panel.empty:
        return spread_panel
    relabelled = spread_panel.iloc[:-1].copy()
    relabelled.index = spread_panel.index[1:]
    return relabelled


def _build_spread_panel(spec, raw_csv_path: Path):
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


def main() -> None:
    SPEC_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_SPEC
    print(f"Loading spec: {SPEC_PATH.name}")

    spec = load_spec(SPEC_PATH)
    raw = load_raw_csv(DATA_PATH, scrub=spec.scrubbing, adjustments=spec.adjustments)

    # Load raw_df directly for benchmark computation (before PITLoader processing)
    raw_df = pd.read_csv(DATA_PATH)

    print(f"Building PIT snapshot (asof={spec.asof})...")
    snap = PITLoader(raw).as_of(pd.Timestamp(spec.asof, tz="UTC"))

    print("Building prepared panel...")
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

    spread_panel = _build_spread_panel(spec, DATA_PATH)

    cost_config = CostConfig(
        spread_bps=spec.cost.spread_bps,
        sharesies_monthly_fee_nzd=spec.cost.sharesies_monthly_fee_nzd,
        sharesies_coverage_nzd=spec.cost.sharesies_coverage_nzd,
        sharesies_excess_bps=spec.cost.sharesies_excess_bps,
    )
    backtest_config = BacktestConfig(
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

    factors = build_factors_from_specs(spec.factors)
    delisting_csv_path = _PYTHON_ROOT / spec.survivorship.delisting_csv_relpath

    print(f"\nRunning rolling walk-forward "
          f"(train={spec.walk_forward.rolling.train_years}y / "
          f"oos={spec.walk_forward.rolling.oos_years}y)...")

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
    oos_returns = wf_result.oos_returns
    print(f"  {len(wf_result.folds)} folds, {len(oos_returns)} OOS months "
          f"({oos_returns.index[0]:%Y-%m} to {oos_returns.index[-1]:%Y-%m})")

    rf = spec.backtest.risk_free_annual
    date_index = oos_returns.index

    # --- Compute benchmarks ---
    print("\nComputing benchmarks...")
    sixty_forty = compute_sixty_forty_returns(raw_df, spec, date_index)
    td_floor = compute_td_floor_returns(spec, date_index)

    # --- Stats ---
    strat_stats = {
        "name": f"{spec.name} (OOS)",
        "sharpe": wf_result.oos_sharpe_raw,
        "mean_annual": _mean_annual(oos_returns),
        "max_dd": wf_result.oos_max_drawdown_observed,
    }
    bm_sixty_forty = compute_benchmark_stats("FNZ 60/40", sixty_forty, rf)
    bm_td = compute_benchmark_stats(f"NZ TD Floor ({spec.benchmarks.td_floor_default:.0%})", td_floor, rf)

    rows = [strat_stats, bm_sixty_forty, bm_td]

    oos_start = oos_returns.index[0].year
    oos_end = oos_returns.index[-1].year
    print(f"\n=== Benchmark Comparison: {spec.name} vs benchmarks (OOS {oos_start}–{oos_end}) ===\n")

    hdr = f"{'':28}  {'Sharpe':>8}  {'MeanAnn%':>9}  {'MaxDD%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        name = r["name"]
        sharpe = r["sharpe"]
        mean_ann = r["mean_annual"]
        max_dd = r["max_dd"]
        dd_str = f"{max_dd:>7.1%}" if not np.isnan(max_dd) else "       -"
        print(f"  {name:<26}  {sharpe:>8.3f}  {mean_ann:>+9.1%}  {dd_str}")

    print()

    # --- Dominance summary ---
    print("=== Dominance Summary ===\n")
    strat_sharpe = strat_stats["sharpe"]
    strat_mean = strat_stats["mean_annual"]
    for bm in [bm_sixty_forty, bm_td]:
        bm_sharpe = bm["sharpe"]
        beats_sharpe = (not np.isnan(bm_sharpe)) and strat_sharpe > bm_sharpe
        beats_return = strat_mean > bm["mean_annual"]
        sharpe_margin = strat_sharpe - bm_sharpe if not np.isnan(bm_sharpe) else float("nan")
        sharpe_bm_str = f"{bm_sharpe:.3f}" if not np.isnan(bm_sharpe) else "n/a (flat)"
        margin_str = f"{sharpe_margin:+.3f}" if not np.isnan(sharpe_margin) else "n/a"
        print(f"  vs {bm['name']}:")
        print(f"    Sharpe: {strat_sharpe:.3f} vs {sharpe_bm_str}  "
              f"(margin={margin_str})  {'BEATS' if beats_sharpe else 'BEATS (flat)' if np.isnan(bm_sharpe) else 'TRAILS'}")
        print(f"    Return: {strat_mean:+.1%}/yr vs {bm['mean_annual']:+.1%}/yr  "
              f"{'BEATS' if beats_return else 'TRAILS'}")
        print()

    # --- Data notes ---
    fnz_rows = raw_df[(raw_df["ticker"] == "FNZ.NZ") & (raw_df["feature"] == "adj_close")]
    if not fnz_rows.empty:
        fnz_start = pd.to_datetime(fnz_rows["timestamp"], unit="ms").min()
        print(f"  Note: FNZ.NZ adj_close history starts {fnz_start:%Y-%m}; "
              f"pre-history 60/40 months filled with 0.")


if __name__ == "__main__":
    main()
