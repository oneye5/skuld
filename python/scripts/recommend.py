"""Generate monthly trade recommendations from a pre-registered spec.

Usage (from python/ directory):
    uv run python scripts/recommend.py --spec configs/strategy-specs/candidates/mom-ar-spread.yaml --holdings tests/fixtures/sharesies_export_2026-04-26.csv --cash tests/fixtures/cash_2026-04-26.yaml --asof 2025-12-31 --raw-csv ../data/data_long.csv --output-dir reports/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Skuld trade recommendations from spec",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Path to strategy spec YAML file",
    )
    p.add_argument(
        "--holdings",
        type=Path,
        required=True,
        help="Path to Sharesies export CSV",
    )
    p.add_argument(
        "--cash",
        type=Path,
        required=True,
        help="Path to cash YAML",
    )
    p.add_argument(
        "--asof",
        type=str,
        required=True,
        help="Rebalance date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--raw-csv",
        type=Path,
        default=None,
        help="Path to data_long.csv (defaults to ../data/data_long.csv)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for recommendations CSV",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    
    # Resolve python root
    python_dir = Path(__file__).resolve().parent.parent
    
    # Default raw_csv path
    if args.raw_csv is None:
        args.raw_csv = python_dir.parent / "data" / "data_long.csv"
    
    if not args.raw_csv.exists():
        print(f"ERROR: data file not found: {args.raw_csv}", file=sys.stderr)
        return 1
    
    if not args.spec.exists():
        print(f"ERROR: spec file not found: {args.spec}", file=sys.stderr)
        return 1
    
    if not args.holdings.exists():
        print(f"ERROR: holdings CSV not found: {args.holdings}", file=sys.stderr)
        return 1
    
    if not args.cash.exists():
        print(f"ERROR: cash YAML not found: {args.cash}", file=sys.stderr)
        return 1
    
    # Parse asof
    try:
        asof = pd.Timestamp(args.asof, tz="UTC")
    except Exception as e:
        print(f"ERROR: invalid asof date '{args.asof}': {e}", file=sys.stderr)
        return 1
    
    print("=" * 60)
    print("  Skuld - Generate Recommendations")
    print("=" * 60)
    print(f"  Spec:     {args.spec}")
    print(f"  Holdings: {args.holdings}")
    print(f"  Cash:     {args.cash}")
    print(f"  As-of:    {asof.date()}")
    print(f"  Data:     {args.raw_csv}")
    print(f"  Output:   {args.output_dir}")
    print("-" * 60)
    
    # Deferred imports for fast --help
    from skuld_portfolio.pipeline.recommend import recommend
    from skuld_portfolio.output.write_recommendations_csv import write_recommendations_csv
    from skuld_research.config.loader import load_spec
    
    try:
        spec = load_spec(args.spec)
        trades, combined_scores, meta = recommend(
            spec_path=args.spec,
            holdings_path=args.holdings,
            cash_yaml_path=args.cash,
            asof=asof,
            raw_csv_path=args.raw_csv,
        )
        
        # Write output
        output_path = args.output_dir / f"recommendations_{asof.date().isoformat()}.csv"
        write_recommendations_csv(trades, spec, meta, output_path, combined_scores=combined_scores)
        
        # Print summary
        print()
        print("Summary:")
        print(f"  Total volume:        ${trades.total_volume_nzd:,.2f} NZD")
        print(f"  Total estimated cost: ${trades.total_estimated_cost_nzd:,.2f} NZD")
        
        action_counts = trades.trades["action"].value_counts().to_dict()
        print(f"  Actions:")
        for action in ["BUY", "SELL", "HOLD", "DEFER"]:
            count = action_counts.get(action, 0)
            print(f"    {action:6s}: {count:3d}")
        
        print()
        print(f"Output written to:")
        print(f"  {output_path}")
        print(f"  {output_path.with_suffix('.meta.json')}")
        
        overrides_log = output_path.parent / f"overrides_log_{asof.date().isoformat()}.csv"
        print(f"  {overrides_log}")
        
        print()
        print("Spec hash:", meta["spec_hash"])
        print()
        
        return 0
    
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
