"""Run a backtest from a pre-registered spec YAML.

Usage (from python/ directory):
    uv run python scripts/backtest.py --spec configs/preregistered/2026-04-26_momentum_only.yaml
    uv run python scripts/backtest.py --spec configs/preregistered/2026-04-26_momentum_only.yaml --raw-csv ../data/data_long.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Skuld backtest from spec YAML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Path to spec YAML file (e.g., configs/preregistered/2026-04-26_momentum_only.yaml)",
    )
    p.add_argument(
        "--raw-csv",
        type=Path,
        default=None,
        help="Path to data_long.csv (defaults to ../data/data_long.csv relative to python/)",
    )
    p.add_argument(
        "--ledger-root",
        type=Path,
        default=None,
        help="Override ledger root directory (defaults to python/trial_ledger)",
    )
    p.add_argument(
        "--no-write-ledger",
        action="store_true",
        help="Skip writing to trial ledger",
    )
    return p.parse_args()


def _get_git_sha() -> str:
    """Get git SHA with -dirty suffix if working tree is dirty."""
    try:
        # Find workspace root (parent of python/)
        python_dir = Path(__file__).resolve().parent.parent
        workspace_dir = python_dir.parent
        
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=workspace_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        
        # Check if dirty
        dirty = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=workspace_dir,
            stderr=subprocess.DEVNULL,
        )
        if dirty != 0:
            sha += "-dirty"
        
        return sha
    except Exception:
        return "unknown"


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
    
    print("=" * 60)
    print("  Skuld - Backtest from Spec")
    print("=" * 60)
    print(f"  Spec:        {args.spec}")
    print(f"  Data:        {args.raw_csv}")
    print(f"  Write ledger: {not args.no_write_ledger}")
    print("-" * 60)
    
    # Deferred imports for fast --help
    from skuld_research.config import load_spec, spec_hash, run_from_spec
    from skuld_research.reporting import build_methodology_report, write_methodology_report
    
    # Load spec
    print("Loading spec...", end=" ", flush=True)
    try:
        spec = load_spec(args.spec)
        h = spec_hash(spec)
        print(f"done [hash: {h[:12]}...]")
    except Exception as e:
        print(f"FAILED", file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    
    # Run backtest
    print("Running backtest...", flush=True)
    try:
        result = run_from_spec(
            spec,
            raw_csv_path=args.raw_csv,
            write_ledger=not args.no_write_ledger,
            ledger_root=args.ledger_root,
        )
        print("Backtest complete.")
    except Exception as e:
        print(f"ERROR during backtest: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    # Build report
    print("Building methodology report...", end=" ", flush=True)
    git_sha = _get_git_sha()
    
    report = build_methodology_report(
        strategy_name=spec.name,
        strategy_two_fold=result.strategy_two_fold or result.strategy_rolling,
        strategy_rolling=result.strategy_rolling,
        benchmarks=result.benchmarks,
        gating=result.gating,
        dominance=result.dominance,
        config_hash=h,
        git_sha=git_sha,
        asof=result.panel.asof,
        panel_coverage_start=result.panel_coverage_start,
        panel_coverage_end=result.panel_coverage_end,
        master_seed=spec.master_seed,
        n_trials_prior=spec.n_trials_prior,
    )
    print("done")
    
    # Write report
    report_dir = python_dir / spec.output.report_dir_relpath
    report_dir.mkdir(parents=True, exist_ok=True)
    
    asof_str = spec.asof.strftime("%Y-%m-%d")
    output_path = report_dir / f"{asof_str}_methodology.md"
    
    print(f"Writing report to {output_path}...", end=" ", flush=True)
    write_methodology_report(report, output_path)
    print("done")
    
    # Summary (ASCII only, no box-drawing characters)
    print("-" * 60)
    print("Summary:")
    print(f"  * Config hash:  {h[:12]}... ({len(h)} chars)")
    print(f"  * Strategy:     {spec.name}")
    print(f"  * OOS Sharpe:   {result.strategy_rolling.oos_sharpe_delisting_adjusted:.3f}")
    print(f"  * Gating:       {'PASS' if result.gating.passes else 'FAIL'}")
    print(f"  * Report:       {output_path}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
