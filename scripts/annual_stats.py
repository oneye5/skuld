"""
Annual stats: year-by-year performance and data quality breakdown for a strategy spec.

Usage (from repo root):
    uv run --project python python scripts/annual_stats.py [path/to/spec.yaml]

If no spec path is given, defaults to mom-s8.yaml.
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
from skuld_research.backtest.metrics import compute_drawdown_series
from skuld_research.costs.model import CostConfig
from skuld_research.costs.spread_estimator import compute_abdi_ranaldo_spread_panel

DATA_PATH = _REPO_ROOT / "data" / "data_long.csv"
_DEFAULT_SPEC = _REPO_ROOT / "python" / "configs" / "strategy-specs" / "candidates" / "mom-s8.yaml"

RF_MONTHLY = 0.0  # risk-free rate per month (spec's risk_free_annual / 12 used below)


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


def _annual_rows(result, rf_monthly: float, panel) -> pd.DataFrame:
    returns = result.returns
    cost_drag = result.cost_drag
    turnover = result.turnover
    n_positions = result.period_n_positions.astype(float)

    rows = []
    for year, grp_idx in returns.groupby(returns.index.year).groups.items():
        yr_ret = returns.loc[grp_idx]
        yr_cost = cost_drag.loc[grp_idx]
        yr_turn = turnover.loc[grp_idx]
        yr_pos = n_positions.loc[grp_idx]

        n = len(yr_ret)
        mu_m = float(yr_ret.mean())
        ann_return = mu_m * 12.0
        ann_vol = float(yr_ret.std(ddof=1)) * np.sqrt(12.0) if n > 1 else float("nan")
        sharpe = ((ann_return - rf_monthly * 12.0) / ann_vol) if (n >= 3 and ann_vol > 1e-12) else float("nan")
        hit_rate = float((yr_ret > 0).mean())

        # Max drawdown within the year: compute drawdown on the year's returns alone
        yr_drawdown = compute_drawdown_series(yr_ret)
        max_dd = float(yr_drawdown.min()) if not yr_drawdown.empty else 0.0

        avg_positions = float(yr_pos.mean())
        avg_turnover = float(yr_turn.mean())

        # Total cost as % of starting NAV: approximate as sum of cost_drag
        # (each cost_drag is cost / nav_before_cost, so summing gives total drag)
        total_cost_pct = float(yr_cost.sum()) * 100.0

        # --- Data quality ---
        # universe_mask: bool DataFrame indexed by rebalance date
        umask = panel.universe_mask
        yr_umask = umask[umask.index.year == year] if not umask.empty else pd.DataFrame()
        universe_size = float(yr_umask.sum(axis=1).mean()) if not yr_umask.empty else float("nan")

        # coverage: fraction of universe tickers with non-NaN returns_monthly
        ret_m = panel.returns_monthly
        yr_ret_m = ret_m[ret_m.index.year == year] if ret_m is not None and not ret_m.empty else pd.DataFrame()
        if not yr_umask.empty and not yr_ret_m.empty:
            coverage_vals = []
            for dt in yr_umask.index:
                if dt not in yr_ret_m.index:
                    continue
                universe_tickers = yr_umask.loc[dt][yr_umask.loc[dt]].index
                if len(universe_tickers) == 0:
                    continue
                row_ret = yr_ret_m.loc[dt, universe_tickers.intersection(yr_ret_m.columns)]
                coverage_vals.append(row_ret.notna().mean())
            coverage_pct = float(np.mean(coverage_vals)) * 100.0 if coverage_vals else float("nan")
        else:
            coverage_pct = float("nan")

        rows.append({
            "year": year,
            "n_months": n,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "hit_rate": hit_rate,
            "max_dd": max_dd,
            "avg_positions": avg_positions,
            "avg_turnover": avg_turnover,
            "total_cost_pct": total_cost_pct,
            "universe_size": universe_size,
            "coverage_pct": coverage_pct,
        })

    return pd.DataFrame(rows).set_index("year")


def _fmt(val, fmt: str, pct: bool = False) -> str:
    if isinstance(val, float) and np.isnan(val):
        return "   NaN"
    if pct:
        return f"{val:{fmt}}"
    return f"{val:{fmt}}"


def print_table(df: pd.DataFrame) -> None:
    header = (
        f"{'Year':>4}  {'N':>3}  {'AnnRet':>7}  {'AnnVol':>7}  {'Sharpe':>7}  "
        f"{'Hit%':>5}  {'MaxDD':>7}  {'AvgPos':>6}  {'AvgTurn':>7}  "
        f"{'Cost%':>6}  {'UnivSz':>6}  {'Cov%':>5}"
    )
    print(header)
    print("-" * len(header))

    for year, row in df.iterrows():
        flag = " (!)" if row["universe_size"] < 10 else "    "
        sharpe_s = f"{row['sharpe']:7.2f}" if not np.isnan(row["sharpe"]) else "    NaN"
        print(
            f"{year:>4}{flag}"
            f"  {int(row['n_months']):>3}"
            f"  {row['ann_return']:>+7.1%}"
            f"  {row['ann_vol']:>7.1%}"
            f"  {sharpe_s}"
            f"  {row['hit_rate']:>5.1%}"
            f"  {row['max_dd']:>+7.1%}"
            f"  {row['avg_positions']:>6.1f}"
            f"  {row['avg_turnover']:>7.1%}"
            f"  {row['total_cost_pct']:>6.2f}"
            f"  {row['universe_size']:>6.1f}"
            f"  {row['coverage_pct']:>5.1f}"
        )


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Summary ===")
    strong = df[df["sharpe"] > 1.0]
    weak = df[df["sharpe"] < 0.0]

    if not strong.empty:
        years = ", ".join(str(y) for y in strong.index.tolist())
        print(f"  Strong years (Sharpe > 1.0):  {years}")
    else:
        print("  Strong years (Sharpe > 1.0):  none")

    if not weak.empty:
        years = ", ".join(str(y) for y in weak.index.tolist())
        print(f"  Weak years   (Sharpe < 0.0):  {years}")
    else:
        print("  Weak years   (Sharpe < 0.0):  none")

    univ_min = df["universe_size"].min()
    univ_max = df["universe_size"].max()
    print(f"  Universe size range:          {univ_min:.0f} – {univ_max:.0f}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", nargs="?", default=str(_DEFAULT_SPEC))
    parser.add_argument("--start-year", type=int, default=2012,
                        help="First year to include in reported table/summary stats (default: 2012)")
    args = parser.parse_args()

    SPEC_PATH = Path(args.spec)
    start_year = args.start_year

    print(f"Loading spec: {SPEC_PATH.name}")
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

    # Build spread panel if spec uses abdi_ranaldo
    spread_panel = _build_spread_panel(spec, DATA_PATH)

    print("Running backtest...")
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
    factors = build_factors_from_specs(spec.factors)
    engine = BacktestEngine(factors=factors, panel=panel, config=config, spread_panel=spread_panel)
    result = engine.run()

    rf_monthly = spec.backtest.risk_free_annual / 12.0

    print(f"\n=== Annual Stats: {SPEC_PATH.name} ===")
    print(f"  Full period: {result.start:%Y-%m} to {result.end:%Y-%m}  ({result.n_periods} months)")
    print(f"  Full-sample Sharpe: {result.sharpe_raw:.3f}  |  Hit rate: {result.hit_rate:.1%}  |  Avg positions: {result.avg_positions:.1f}")
    print()

    df_all = _annual_rows(result, rf_monthly, panel)
    df = df_all[df_all.index >= start_year]

    print(f"  (Reporting window: {start_year}+ only — full history used for backtest warmup)\n")
    print_table(df)
    print_summary(df)

    # Restricted-window aggregate stats
    valid = df.dropna(subset=["sharpe"])
    if not valid.empty:
        # Collect monthly returns for the restricted window
        restricted_returns = result.returns[result.returns.index.year >= start_year]
        if len(restricted_returns) > 1:
            mu_ann = restricted_returns.mean() * 12.0
            vol_ann = restricted_returns.std(ddof=1) * np.sqrt(12.0)
            sharpe_restricted = (mu_ann - spec.backtest.risk_free_annual) / vol_ann if vol_ann > 1e-12 else float("nan")
            print(f"\n=== {start_year}–2026 Aggregate ===")
            print(f"  Mean annual return: {mu_ann:+.1%}")
            print(f"  Annual vol:         {vol_ann:.1%}")
            print(f"  Sharpe:             {sharpe_restricted:.3f}")


if __name__ == "__main__":
    main()
