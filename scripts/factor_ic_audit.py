"""
Factor IC audit: measures cross-sectional rank IC per factor and runs full backtest.

Usage (from repo root):
    uv run --project python python scripts/factor_ic_audit.py [path/to/spec.yaml]

If no spec path is given, defaults to mom-s6.yaml.

Outputs:
    - Per-factor: mean IC, ICIR, hit rate (fraction of dates with IC > 0), n dates
    - Full backtest summary for the given spec
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python" / "src"))

from skuld_research.config.loader import load_spec
from skuld_research.config.factors import build_factors_from_specs
from skuld_research.data.csv_loader import load_raw_csv
from skuld_research.data.pit_loader import PITLoader
from skuld_research.data.prepared_panel import build_prepared_panel
from skuld_research.backtest.engine import BacktestConfig, BacktestEngine
from skuld_research.costs.model import CostConfig

DATA_PATH = _REPO_ROOT / "data" / "data_long.csv"
_DEFAULT_SPEC = _REPO_ROOT / "python" / "configs" / "strategy-specs" / "candidates" / "mom-s6.yaml"


def rank_ic(scores: pd.Series, fwd_returns: pd.Series) -> float | None:
    common = scores.dropna().index.intersection(fwd_returns.dropna().index)
    if len(common) < 5:
        return None
    ic, _ = spearmanr(scores.loc[common], fwd_returns.loc[common])
    return float(ic) if not np.isnan(ic) else None


def compute_factor_ics(panel, rebalance_dates: list, factor, fwd_returns_monthly: pd.DataFrame) -> list[float]:
    ics = []
    for t in rebalance_dates:
        if t in panel.universe_mask.index:
            mask = panel.universe_mask.loc[t]
        else:
            print(f"  Warning: date {t} not in universe_mask, skipping")
            continue
        universe = mask[mask].index.tolist()
        if not universe:
            continue

        scores = factor.score(panel, t, universe)
        if scores is None or scores.empty:
            continue

        date_idx = fwd_returns_monthly.index.searchsorted(t)
        if date_idx + 1 >= len(fwd_returns_monthly):
            continue
        fwd_date = fwd_returns_monthly.index[date_idx + 1]
        fwd_ret = fwd_returns_monthly.loc[fwd_date].dropna()

        ic = rank_ic(scores, fwd_ret)
        if ic is not None:
            ics.append(ic)
    return ics


def print_ic_summary(name: str, ics: list[float]) -> None:
    if not ics:
        print(f"  {name:<30} no valid IC dates")
        return
    arr = np.array(ics)
    mean_ic = arr.mean()
    icir = mean_ic / arr.std() if arr.std() > 1e-9 else float("nan")
    hit = (arr > 0).mean()
    print(f"  {name:<30} mean_IC={mean_ic:+.4f}  ICIR={icir:+.3f}  hit={hit:.1%}  n={len(arr)}")


def main() -> None:
    SPEC_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_SPEC
    print("Loading data...")
    spec = load_spec(SPEC_PATH)
    raw = load_raw_csv(DATA_PATH, scrub=spec.scrubbing, adjustments=spec.adjustments)

    print(f"Building PIT snapshot (asof={spec.asof})...")
    snap = PITLoader(raw).as_of(pd.Timestamp(spec.asof, tz="UTC"))

    print(f"Building prepared panel ({SPEC_PATH.name})...")
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

    rebalance_dates = panel.universe_mask.index.tolist()
    fwd_returns_monthly = panel.returns_monthly

    factors_named = [(fs.kind, f) for fs, f in zip(spec.factors, build_factors_from_specs(spec.factors))]

    print("\n=== Factor IC Audit ===")
    for name, factor in factors_named:
        print(f"  Scoring {name}...")
        ics = compute_factor_ics(panel, rebalance_dates, factor, fwd_returns_monthly)
        print_ic_summary(name, ics)

    print(f"\n=== Backtest ({SPEC_PATH.name}) ===")
    cost_config = CostConfig(
        spread_bps=spec.cost.spread_bps,
        sharesies_monthly_fee_nzd=spec.cost.sharesies_monthly_fee_nzd,
        sharesies_coverage_nzd=spec.cost.sharesies_coverage_nzd,
        sharesies_excess_bps=spec.cost.sharesies_excess_bps,
    )
    config = BacktestConfig(
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
    )
    factor_instances = [f for _, f in factors_named]
    engine = BacktestEngine(factors=factor_instances, panel=panel, config=config)
    result = engine.run()
    print(f"  Period:              {result.start:%Y-%m} to {result.end:%Y-%m}  ({result.n_periods} months)")
    print(f"  Sharpe (raw):        {result.sharpe_raw:.3f}")
    print(f"  Sharpe (400bps hc):  {result.sharpe_flat_haircut:.3f}")
    print(f"  Calmar:              {result.calmar_ratio:.3f}")
    print(f"  Hit rate:            {result.hit_rate:.1%}")
    print(f"  Avg positions:       {result.avg_positions:.1f}")


if __name__ == "__main__":
    main()
