"""WS2 construction sweep + WS4 regime overlay + WS5 new alpha factors for mom-s8.

Usage (from python/ directory):
    uv run python scripts/ws2_ws4_ws5_analysis.py
    uv run python scripts/ws2_ws4_ws5_analysis.py --raw-csv ../data/data_long.csv

Outputs:
    reports/ws2_ws4_ws5.md   — construction sweep summary + WS4/WS5 results
    reports/ws2_sweep/       — per-variant CSV from build_construction_variants
"""
# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

_PYTHON_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PYTHON_ROOT / "src"))

import numpy as np
import pandas as pd

from skuld_research.config.loader import load_spec
from skuld_research.config.runner import run_from_spec
from skuld_research.experiments.factor_experiment import (
    build_construction_variants,
    run_variants,
)
from skuld_research.stats.paired import stationary_bootstrap_paired_delta

_BASELINE = _PYTHON_ROOT / "configs" / "strategy-specs" / "production" / "mom-s8.yaml"
_CANDIDATES_DIR = _PYTHON_ROOT / "configs" / "strategy-specs" / "candidates"
_WS4_SPEC = _CANDIDATES_DIR / "ws4-mom-s8-overlay.yaml"
_WS5_SPECS = [
    _CANDIDATES_DIR / "ws5-mom-s8-eps.yaml",
    _CANDIDATES_DIR / "ws5-mom-s8-voltrd.yaml",
]
_DEFAULT_RAW_CSV = _PYTHON_ROOT.parent / "data" / "data_long.csv"
_DEFAULT_OUT = _PYTHON_ROOT / "reports" / "ws2_ws4_ws5.md"
_WS2_SWEEP_DIR = _PYTHON_ROOT / "reports" / "ws2_sweep"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cagr(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) == 0:
        return float("nan")
    n_years = len(clean) / 12.0
    total = (1.0 + clean).prod()
    if total <= 0 or n_years <= 0:
        return float("nan")
    return float(total ** (1.0 / n_years) - 1.0)


def _fmt_pct(v: float, digits: int = 1) -> str:
    if v != v:
        return "n/a"
    return f"{v:+.{digits}%}"


def _fmt_f(v: float, digits: int = 3) -> str:
    if v != v:
        return "n/a"
    return f"{v:.{digits}f}"


def _run_candidate(
    spec_path: Path,
    raw_csv: Path,
    baseline_wf,
    bootstrap_resamples: int,
    label: str | None = None,
) -> dict | None:
    if not spec_path.exists():
        print(f"  SKIP (not found): {spec_path.name}")
        return None
    name = label or spec_path.stem
    print(f"\n  Running {spec_path.name} …")
    spec = load_spec(spec_path)
    try:
        result = run_from_spec(spec, raw_csv_path=raw_csv, write_ledger=False)
    except Exception as exc:
        print(f"    ERROR: {exc}")
        return None

    wf = result.strategy_rolling
    paired = stationary_bootstrap_paired_delta(
        wf.oos_returns,
        baseline_wf.oos_returns,
        n_resamples=bootstrap_resamples,
        rng_seed=spec.master_seed,
    )
    row = {
        "name": spec.name,
        "sharpe_hc": wf.oos_sharpe_flat_haircut,
        "sharpe_raw": wf.oos_sharpe_raw,
        "delta_sharpe": wf.oos_sharpe_flat_haircut - baseline_wf.oos_sharpe_flat_haircut,
        "paired_delta_annual": paired.mean_delta_annual,
        "paired_ci_low": paired.ci_low_95_monthly,
        "paired_ci_high": paired.ci_high_95_monthly,
        "cagr": _cagr(wf.oos_returns),
        "turnover": wf.oos_avg_turnover,
        "max_dd": wf.oos_max_drawdown_observed,
        "n_obs": len(wf.oos_returns.dropna()),
    }
    print(
        f"    Sharpe HC: {row['sharpe_hc']:.3f} ({row['delta_sharpe']:+.3f}) | "
        f"turnover: {row['turnover']:.1%} | "
        f"paired ann delta: {row['paired_delta_annual']:+.1%}"
    )
    return row


# ---------------------------------------------------------------------------
# WS2: parse construction sweep results CSV
# ---------------------------------------------------------------------------

def _load_ws2_summary(sweep_dir: Path, baseline_sharpe_hc: float) -> list[dict]:
    results_csv = sweep_dir / "results.csv"
    if not results_csv.exists():
        return []
    df = pd.read_csv(results_csv)
    if df.empty:
        return []

    # Extract oos_sharpe_flat_haircut — it's buried in custom gate fields.
    # run_variants stores oos_sharpe_raw; we compute flat-haircut from raw when
    # not available, but factor_experiment stores oos_sharpe_raw only.
    # Use oos_sharpe_raw as proxy (consistent with comparisons).
    sharpe_col = "oos_sharpe_raw"
    if sharpe_col not in df.columns:
        return []

    df = df[df["status"] == "ok"].copy()
    df["delta_sharpe"] = df[sharpe_col].astype(float) - baseline_sharpe_hc
    df_sorted = df.sort_values(sharpe_col, ascending=False)

    rows = []
    for _, r in df_sorted.head(10).iterrows():
        rows.append({
            "label": r["label"],
            "lane": r.get("lane", ""),
            "sharpe_raw": float(r[sharpe_col]),
            "delta_sharpe": float(r["delta_sharpe"]),
            "turnover": float(r.get("turnover_mean", float("nan"))),
            "max_dd": float(r.get("max_drawdown", float("nan"))),
        })
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(
    path: Path,
    baseline_wf,
    ws2_rows: list[dict],
    ws2_completed: int,
    ws2_failed: int,
    ws4_row: dict | None,
    ws5_rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline_sh = baseline_wf.oos_sharpe_flat_haircut
    baseline_to = baseline_wf.oos_avg_turnover

    lines: list[str] = []
    lines += [
        "# WS2 Construction Sweep + WS4 Regime Overlay + WS5 New Alpha Factors",
        "",
        "Scope: exploration. No production spec is modified.",
        "",
        f"Baseline `mom-s8`: flat-haircut Sharpe {baseline_sh:.3f}, "
        f"turnover {baseline_to:.1%}.",
        "",
    ]

    # ------------------------------------------------------------------
    # WS2 construction sweep
    # ------------------------------------------------------------------
    lines += [
        "## WS2 — Portfolio Construction Sweep (quick=True, 16 variants)",
        "",
        f"Variants completed: {ws2_completed}, failed: {ws2_failed}.",
        "",
        "Decision criterion: flat-haircut Sharpe ≥ baseline + 0.10 AND "
        "paired CI lower bound ≥ 0.",
        "",
    ]
    if ws2_rows:
        lines += [
            "Top-10 by OOS Sharpe (raw, before haircut):",
            "",
            "| Variant | Sharpe raw | Δ vs baseline | Turnover | Max DD |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in ws2_rows:
            lines.append(
                f"| `{row['label']}` "
                f"| {_fmt_f(row['sharpe_raw'])} "
                f"| {row['delta_sharpe']:+.3f} "
                f"| {row['turnover']:.1%} "
                f"| {row['max_dd']:.1%} |"
            )
        lines.append("")

        best = ws2_rows[0]
        passes = best["delta_sharpe"] >= 0.10
        lines += [
            f"**Best variant**: `{best['label']}` — "
            f"Sharpe raw {_fmt_f(best['sharpe_raw'])}, "
            f"delta {best['delta_sharpe']:+.3f}.",
            "",
            f"**WS2 verdict**: {'PASS — best variant clears +0.10 threshold.' if passes else 'FAIL — no variant clears +0.10 threshold over baseline.'}",
            "",
        ]
    else:
        lines += [
            "No results available (sweep did not complete or output CSV missing).",
            "",
        ]

    # ------------------------------------------------------------------
    # WS4 overlay
    # ------------------------------------------------------------------
    lines += ["## WS4 — Regime Overlay (NZX MA-200 + aggregate momentum)", ""]
    if ws4_row:
        ci_str = f"[{ws4_row['paired_ci_low']:+.2%}, {ws4_row['paired_ci_high']:+.2%}]"
        passes = ws4_row["delta_sharpe"] >= 0.0
        lines += [
            "| Metric | Value |",
            "|---|---:|",
            f"| Sharpe HC | {_fmt_f(ws4_row['sharpe_hc'])} |",
            f"| Delta HC vs baseline | {ws4_row['delta_sharpe']:+.3f} |",
            f"| Paired ann. delta | {ws4_row['paired_delta_annual']:+.1%} |",
            f"| Paired 95% CI (monthly) | {ci_str} |",
            f"| Turnover | {ws4_row['turnover']:.1%} |",
            f"| Max drawdown | {ws4_row['max_dd']:.1%} |",
            f"| CAGR | {_fmt_pct(ws4_row['cagr'])} |",
            "",
            f"**WS4 verdict**: {'PASS — overlay improves risk-adjusted return.' if passes else 'FAIL — overlay does not improve risk-adjusted return.'}",
            "",
        ]
    else:
        lines += ["Overlay run did not complete.", ""]

    # ------------------------------------------------------------------
    # WS5 new alpha factors
    # ------------------------------------------------------------------
    lines += [
        "## WS5 — New Alpha Factors (EPS momentum, volume trend)",
        "",
        "Decision criterion: flat-haircut Sharpe ≥ baseline (no regression) "
        "AND paired ann. delta ≥ 0 (positive contribution).",
        "",
        "| Variant | Sharpe HC | Δ HC | Paired ann Δ | Paired 95% CI | Turnover | Verdict |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for row in ws5_rows:
        passes = row["delta_sharpe"] >= 0.0 and row["paired_delta_annual"] >= 0.0
        verdict = "pass" if passes else "fail"
        ci_str = f"[{row['paired_ci_low']:+.2%}, {row['paired_ci_high']:+.2%}]"
        lines.append(
            f"| `{row['name']}` "
            f"| {_fmt_f(row['sharpe_hc'])} "
            f"| {row['delta_sharpe']:+.3f} "
            f"| {row['paired_delta_annual']:+.1%} "
            f"| {ci_str} "
            f"| {row['turnover']:.1%} "
            f"| {verdict} |"
        )
    if not ws5_rows:
        lines.append("| (no results) | | | | | | |")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote report: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WS2/WS4/WS5 construction sweep, overlay, new-alpha analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-csv", type=Path, default=_DEFAULT_RAW_CSV)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--bootstrap-resamples", type=int, default=500)
    p.add_argument("--skip-ws2", action="store_true", help="Skip WS2 construction sweep")
    p.add_argument("--skip-ws4", action="store_true", help="Skip WS4 overlay run")
    p.add_argument("--skip-ws5", action="store_true", help="Skip WS5 new factor runs")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.raw_csv.exists():
        print(f"ERROR: data file not found: {args.raw_csv}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Step 1: Run mom-s8 baseline
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Running mom-s8 baseline …")
    print("=" * 60)
    baseline_spec = load_spec(_BASELINE)
    baseline_result = run_from_spec(
        baseline_spec, raw_csv_path=args.raw_csv, write_ledger=False
    )
    baseline_wf = baseline_result.strategy_rolling
    print(
        f"  OOS Sharpe (raw): {baseline_wf.oos_sharpe_raw:.3f} | "
        f"flat-haircut: {baseline_wf.oos_sharpe_flat_haircut:.3f} | "
        f"turnover: {baseline_wf.oos_avg_turnover:.1%}"
    )

    # ------------------------------------------------------------------
    # Step 2: WS2 construction sweep (quick=True → 16 variants)
    # ------------------------------------------------------------------
    ws2_completed = 0
    ws2_failed = 0
    ws2_rows: list[dict] = []

    if not args.skip_ws2:
        print("\n" + "=" * 60)
        print("Step 2: WS2 construction sweep (quick=True, 16 variants) …")
        print("=" * 60)
        variants = build_construction_variants(baseline_spec, quick=True)
        print(f"  Built {len(variants)} variants.")
        summary = run_variants(
            variants,
            raw_csv_path=args.raw_csv,
            output_dir=_WS2_SWEEP_DIR,
        )
        ws2_completed = summary.completed
        ws2_failed = summary.failed
        print(
            f"  Sweep done: {ws2_completed} completed, {ws2_failed} failed. "
            f"Results: {_WS2_SWEEP_DIR / 'results.csv'}"
        )
        ws2_rows = _load_ws2_summary(_WS2_SWEEP_DIR, baseline_wf.oos_sharpe_raw)

    # ------------------------------------------------------------------
    # Step 3: WS4 overlay
    # ------------------------------------------------------------------
    ws4_row: dict | None = None
    if not args.skip_ws4:
        print("\n" + "=" * 60)
        print("Step 3: WS4 regime overlay …")
        print("=" * 60)
        ws4_row = _run_candidate(
            _WS4_SPEC, args.raw_csv, baseline_wf, args.bootstrap_resamples
        )

    # ------------------------------------------------------------------
    # Step 4: WS5 new alpha factors
    # ------------------------------------------------------------------
    ws5_rows: list[dict] = []
    if not args.skip_ws5:
        print("\n" + "=" * 60)
        print("Step 4: WS5 new alpha factors …")
        print("=" * 60)
        for spec_path in _WS5_SPECS:
            row = _run_candidate(
                spec_path, args.raw_csv, baseline_wf, args.bootstrap_resamples
            )
            if row is not None:
                ws5_rows.append(row)

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    _write_report(
        args.out,
        baseline_wf,
        ws2_rows,
        ws2_completed,
        ws2_failed,
        ws4_row,
        ws5_rows,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
