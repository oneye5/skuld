"""CLI script to run the pipeline audit on NZX data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Skuld NZX pipeline audit.")
    parser.add_argument("--data-path", required=True, help="Path to data_long.csv")
    parser.add_argument("--delisting-csv", default=None, help="Path to NZX delisting CSV (optional)")
    parser.add_argument("--asof", default="2025-12-31", help="As-of date (ISO format, default: 2025-12-31)")
    args = parser.parse_args()

    from skuld_research.data.csv_loader import load_raw_csv
    from skuld_research.data.pit_loader import PITLoader
    from skuld_research.data.prepared_panel import build_prepared_panel
    from skuld_research.diagnostics.audit import audit_pipeline

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

    print("Running audit ...")
    report = audit_pipeline(panel, delisting_csv_path=args.delisting_csv)

    # Print human-readable summary
    print("\n" + "=" * 60)
    print("  SKULD PIPELINE AUDIT REPORT")
    print("=" * 60)

    pit_status = "PASS" if report.pit_compliant else "FAIL (DATA LEAKAGE DETECTED)"
    print(f"\n[PIT Compliance]")
    print(f"  Status              : {pit_status}")
    print(f"  Max return date     : {report.pit_max_return_date.date()}")
    print(f"  As-of date          : {report.pit_asof.date()}")

    align_status = "PASS" if report.rebalance_dates_aligned else "WARN"
    print(f"\n[Timestamp Alignment]")
    print(f"  Status              : {align_status}")
    print(f"  Rebalance dates     : {report.n_rebalance_dates}")
    print(f"  Misaligned dates    : {report.n_misaligned_dates}")

    print(f"\n[Sector Coverage]")
    print(f"  Total tickers       : {report.n_tickers_total}")
    print(f"  Known sector        : {report.n_tickers_known_sector}")
    print(f"  Known sector frac   : {report.frac_known_sector:.1%}")

    print(f"\n[Survivorship Coverage]")
    print(f"  Tickers in panel    : {report.n_tickers_in_panel}")
    print(f"  Tickers in delist   : {report.n_tickers_in_delisting_csv}")
    print(f"  Delist coverage     : {report.delisting_csv_coverage_frac:.1%}")

    print(f"\n[Data Gaps (Returns)]")
    print(f"  Mean NaN fraction   : {report.mean_nan_frac_returns:.1%}")
    print(f"  Max NaN fraction    : {report.max_nan_frac_returns:.1%}")
    print(f"  Worst ticker        : {report.worst_nan_ticker}")

    print(f"\n[Corporate Actions]")
    print(f"  Events              : {report.n_corporate_action_events}")
    print(f"  Tickers affected    : {report.n_corporate_action_tickers}")

    print("\n" + "=" * 60)

    if not report.pit_compliant:
        print("ERROR: PIT compliance check failed — data leakage detected.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
