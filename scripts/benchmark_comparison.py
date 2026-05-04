"""Benchmark comparison using the canonical research runner."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python" / "src"))

from skuld_research.config.loader import load_spec
from skuld_research.config.runner import run_from_spec

DATA_PATH = _REPO_ROOT / "data" / "data_long.csv"
_DEFAULT_SPEC = _REPO_ROOT / "python" / "configs" / "strategy-specs" / "candidates" / "mom-s8.yaml"


def _ascii_safe(text: str) -> str:
    return text.replace("≤", "<=").replace("≥", ">=")


def _mean_annual(returns) -> float:
    return float(returns.mean() * 12.0)


def main() -> None:
    spec_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_SPEC
    print(f"Loading spec: {spec_path.name}")

    spec = load_spec(spec_path)
    run_result = run_from_spec(spec, raw_csv_path=DATA_PATH, write_ledger=False)
    strategy = run_result.strategy_rolling

    rows = [
        {
            "name": f"{spec.name} (OOS)",
            "sharpe": strategy.oos_sharpe_raw,
            "mean_annual": _mean_annual(strategy.oos_returns),
            "max_dd": strategy.oos_max_drawdown_observed,
        }
    ]
    for bench in run_result.benchmarks:
        rows.append(
            {
                "name": bench.name,
                "sharpe": bench.wf_rolling.oos_sharpe_raw,
                "mean_annual": _mean_annual(bench.wf_rolling.oos_returns),
                "max_dd": bench.wf_rolling.oos_max_drawdown_observed,
            }
        )

    oos_start = strategy.oos_returns.index[0].year
    oos_end = strategy.oos_returns.index[-1].year
    print(f"\n=== Benchmark Comparison: {spec.name} vs benchmarks (OOS {oos_start}-{oos_end}) ===\n")

    hdr = f"{'':28}  {'Sharpe':>8}  {'MeanAnn%':>9}  {'MaxDD%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        dd_str = f"{row['max_dd']:>7.1%}" if not np.isnan(row["max_dd"]) else "       -"
        print(
            f"  {row['name']:<26}  {row['sharpe']:>8.3f}  "
            f"{row['mean_annual']:>+9.1%}  {dd_str}"
        )

    print("\n=== Dominance Summary ===\n")
    strat_sharpe = rows[0]["sharpe"]
    strat_mean = rows[0]["mean_annual"]
    for row in rows[1:]:
        bm_sharpe = row["sharpe"]
        beats_sharpe = (not np.isnan(bm_sharpe)) and strat_sharpe > bm_sharpe
        beats_return = strat_mean > row["mean_annual"]
        sharpe_margin = strat_sharpe - bm_sharpe if not np.isnan(bm_sharpe) else float("nan")
        sharpe_bm_str = f"{bm_sharpe:.3f}" if not np.isnan(bm_sharpe) else "n/a"
        margin_str = f"{sharpe_margin:+.3f}" if not np.isnan(sharpe_margin) else "n/a"
        print(f"  vs {row['name']}:")
        print(
            f"    Sharpe: {strat_sharpe:.3f} vs {sharpe_bm_str}  "
            f"(margin={margin_str})  {'BEATS' if beats_sharpe else 'TRAILS'}"
        )
        print(
            f"    Return: {strat_mean:+.1%}/yr vs {row['mean_annual']:+.1%}/yr  "
            f"{'BEATS' if beats_return else 'TRAILS'}"
        )
        dominance_key = f"dominance_{row['name']}"
        if dominance_key in run_result.gating.bars:
            passed, reason = run_result.gating.bars[dominance_key]
            status = "PASS" if passed else "FAIL"
            print(f"    Dominance gate: {status}  {_ascii_safe(reason)}")
        print()

    print("=== Canonical Gating ===\n")
    for bar_name, (passed, reason) in run_result.gating.bars.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {bar_name:<24} {status:>4}  {_ascii_safe(reason)}")

    print()
    for bench in run_result.benchmarks:
        notes = "; ".join(bench.notes) if bench.notes else "none"
        print(
            f"  Note: {bench.name} coverage {bench.coverage_start:%Y-%m} to "
            f"{bench.coverage_end:%Y-%m}; {notes}."
        )


if __name__ == "__main__":
    main()
