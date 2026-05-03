"""CLI script to compare multiple factors by IC, decay, and pairwise IC-series correlation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


FACTOR_REGISTRY_NAMES = {"momentum", "dividend_yield"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Skuld factor IC comparison.")
    parser.add_argument("--data-path", required=True, help="Path to data_long.csv")
    parser.add_argument(
        "--asof", default="2025-12-31", help="As-of date (ISO format, default: 2025-12-31)"
    )
    parser.add_argument(
        "--factors",
        default="momentum,dividend_yield",
        help="Comma-separated factor names (default: momentum,dividend_yield)",
    )
    args = parser.parse_args()

    requested = [f.strip() for f in args.factors.split(",")]
    unsupported = [f for f in requested if f not in FACTOR_REGISTRY_NAMES]
    if unsupported:
        print(f"Error: unsupported factor(s): {', '.join(unsupported)}")
        print(f"Supported: {', '.join(sorted(FACTOR_REGISTRY_NAMES))}")
        sys.exit(1)

    from skuld_research.data.csv_loader import load_raw_csv
    from skuld_research.data.pit_loader import PITLoader
    from skuld_research.data.prepared_panel import build_prepared_panel
    from skuld_research.diagnostics.factor_comparison import compare_factors
    from skuld_research.factors.dividend_yield import DividendYieldFactor
    from skuld_research.factors.momentum import MomentumFactor

    FACTOR_REGISTRY = {
        "momentum": MomentumFactor(),
        "dividend_yield": DividendYieldFactor(),
    }

    factors = {name: FACTOR_REGISTRY[name] for name in requested}

    asof = pd.Timestamp(args.asof, tz="UTC")

    print(f"Loading data from {args.data_path} ...")
    raw = load_raw_csv(Path(args.data_path))

    print(f"Building PIT snapshot as-of {asof} ...")
    loader = PITLoader(raw)
    snap = loader.as_of(asof)

    print("Building PreparedPanel ...")
    panel = build_prepared_panel(
        snap,
        min_adv_dollars=100_000,
        min_market_cap_nzd=10_000_000,
        min_history_days=252,
        adv_window=60,
        mc_ffill_days=5,
        nzx_only=True,
        rebalance_freq="ME",
        anomaly_filter=None,
    )

    print("Comparing factors ...")
    report = compare_factors(factors, panel, min_cross_section=3)

    print("\n" + "=" * 65)
    print("  FACTOR IC COMPARISON REPORT")
    print("=" * 65)

    header = f"{'Factor':<20} {'IC Mean':>10} {'IC IR':>10} {'t-stat':>10} {'Peak H':>8}"
    print("\n" + header)
    print("-" * 65)
    for name in report.factor_names:
        ic = report.ic_reports[name]
        dr = report.decay_reports[name]
        ic_mean = f"{ic.ic_mean:.4f}" if not _isnan(ic.ic_mean) else "   NaN"
        ic_ir = f"{ic.ic_ir:.4f}" if not _isnan(ic.ic_ir) else "   NaN"
        t_stat = f"{ic.t_stat:.4f}" if not _isnan(ic.t_stat) else "   NaN"
        peak_h = str(dr.peak_horizon) if dr.peak_horizon is not None else "  N/A"
        print(f"{name:<20} {ic_mean:>10} {ic_ir:>10} {t_stat:>10} {peak_h:>8}")

    print("\n[IC Series Correlation Matrix]")
    print(report.ic_series_corr.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n[Redundant Pairs]")
    if report.redundant_pairs:
        for a, b in report.redundant_pairs:
            corr_val = report.ic_series_corr.loc[a, b]
            print(f"  {a} <-> {b}  (|corr| = {abs(corr_val):.3f})")
    else:
        print("  None detected.")


def _isnan(val) -> bool:
    import math
    try:
        return math.isnan(val)
    except (TypeError, ValueError):
        return True


if __name__ == "__main__":
    main()
