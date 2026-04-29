"""Generate signal-level diagnostics for momentum factor.

Usage (from python/ directory):
    uv run python scripts/diagnostics.py
    uv run python scripts/diagnostics.py --asof 2026-01-01
    uv run python scripts/diagnostics.py --out reports/

Requires: data/data_long.csv to exist at the workspace root.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── resolve paths ────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent  # python/scripts/
_PYTHON_DIR = _SCRIPT_DIR.parent  # python/
_WORKSPACE_DIR = _PYTHON_DIR.parent  # skuld/
_DATA_PATH = _WORKSPACE_DIR / "data" / "data_long.csv"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Skuld signal diagnostics for momentum",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--asof",
        default="2026-01-01",
        help="Snapshot date (YYYY-MM-DD). Panel is built PIT-safe up to this date.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_PYTHON_DIR / "reports",
        help="Output directory for markdown report.",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=_DATA_PATH,
        help="Path to data_long.csv.",
    )
    return p.parse_args()


def _hr(char: str = "─", width: int = 60) -> str:
    return char * width


def main() -> int:
    args = _parse_args()

    if not args.data.exists():
        print(f"ERROR: data file not found: {args.data}", file=sys.stderr)
        print("  Run the Java ingestion pipeline first, or pass --data <path>.", file=sys.stderr)
        return 1

    print(_hr("═"))
    print("  Skuld — Signal Diagnostics")
    print(_hr("═"))
    print(f"  Data:     {args.data}")
    print(f"  Snapshot: {args.asof}")
    print(f"  Output:   {args.out}")
    print(_hr())

    # ── imports (deferred so --help is instant) ──────────────────────────────
    from skuld_research.data.csv_loader import load_raw_csv
    from skuld_research.data.pit_loader import PITLoader
    from skuld_research.data.prepared_panel import build_prepared_panel
    from skuld_research.factors.momentum import MomentumFactor
    from skuld_research.backtest.engine import BacktestConfig, BacktestEngine
    from skuld_research.diagnostics.panels import (
        score_panel,
        quintile_spread_returns,
        market_proxy_returns,
    )
    from skuld_research.diagnostics.ic import ranking_ic
    from skuld_research.diagnostics.decay import alpha_decay
    from skuld_research.diagnostics.decomposition import factor_decomposition
    from skuld_research.diagnostics.report import write_diagnostics_report

    print("Loading data...")
    raw = load_raw_csv(args.data)

    print("Building PIT snapshot...")
    asof_ts = pd.Timestamp(args.asof, tz="UTC")
    snap = PITLoader(raw).as_of(asof_ts)

    print("Building prepared panel...")
    panel = build_prepared_panel(snap)

    print(f"Panel coverage: {len(panel.universe_mask)} rebalance dates, "
          f"{len(panel.returns_monthly.columns)} tickers")

    print("Scoring momentum factor...")
    momentum = MomentumFactor()
    scores = score_panel(momentum, panel)

    print("Computing IC at horizon=1...")
    ic = ranking_ic(
        scores,
        panel.returns_monthly,
        horizon_months=1,
        factor_name="momentum",
        min_cross_section=10,
    )

    print(f"  IC mean: {ic.ic_mean:.4f}, t-stat: {ic.t_stat_newey_west:.4f}")

    print("Computing alpha decay...")
    decay = alpha_decay(
        scores,
        panel.returns_monthly,
        horizons=(1, 2, 3, 6, 12),
        factor_name="momentum",
        min_cross_section=10,
    )

    print(f"  Peak horizon: {decay.peak_horizon} months")

    print("Computing factor decomposition...")
    # Run the actual momentum backtest to obtain strategy returns (NOT the raw
    # factor spread). The decomposition then asks: does the live strategy add
    # alpha beyond simple exposure to its own component factors?
    bt = BacktestEngine(
        factors=[momentum],
        panel=panel,
        config=BacktestConfig(),
    ).run()
    strategy_ret = bt.returns

    # Component factor return: long-short top-minus-bottom quintile of momentum.
    momentum_spread = quintile_spread_returns(scores, panel.returns_monthly)
    market_ret = market_proxy_returns(panel)

    # Size factor: long-short small-minus-big monthly mcap quintile spread.
    mcap_month_end = panel.market_cap.resample("BME").last()
    size_score = -np.log(mcap_month_end.replace(0.0, np.nan))
    size_spread = quintile_spread_returns(size_score, panel.returns_monthly)

    decomp = factor_decomposition(
        strategy_returns=strategy_ret,
        market_returns=market_ret,
        factor_returns_dict={"size_lmh": size_spread, "momentum": momentum_spread},
    )

    print(f"  Market beta: {decomp.coefficients.get('market', float('nan')):.4f}")
    print(f"  Momentum beta: {decomp.coefficients.get('momentum', float('nan')):.4f}")
    print(f"  Residual alpha: {decomp.residual_alpha_annualised:.4f} (annualized)")

    print("Writing report...")
    args.out.mkdir(parents=True, exist_ok=True)
    asof_str = asof_ts.strftime("%Y-%m-%d")
    out_path = args.out / f"{asof_str}_diagnostics_momentum.md"
    write_diagnostics_report(ic, decay, decomp, out_path)

    print(f"\n✓ Report written to: {out_path}")
    print(_hr("═"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
