"""Run the Phase 2 exploration candidate funnel against mom-s8."""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PYTHON_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PYTHON_ROOT / "src"))

from skuld_research.config.loader import load_spec  # noqa: E402
from skuld_research.config.runner import run_from_spec  # noqa: E402
from skuld_research.stats.paired import stationary_bootstrap_paired_delta  # noqa: E402


DEFAULT_BASELINE = _PYTHON_ROOT / "configs" / "strategy-specs" / "production" / "mom-s8.yaml"
DEFAULT_CANDIDATE_DIR = _PYTHON_ROOT / "configs" / "strategy-specs" / "candidates"
DEFAULT_RAW_CSV = _PYTHON_ROOT.parent / "data" / "data_long.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 exploration candidates and write a Markdown summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument(
        "--out",
        type=Path,
        default=_PYTHON_ROOT / "reports" / "phase2_exploration.md",
    )
    parser.add_argument("--pattern", default="phase2-*.yaml")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    return parser.parse_args()


def _factor_summary(spec) -> str:
    return ", ".join(factor.kind for factor in spec.factors)


def _mean_annual(returns) -> float:
    return float(returns.mean() * 12.0)


def _recommend(row: dict[str, object]) -> str:
    if bool(row["incremental_bar"]):
        return "shortlist-review"
    if float(row["delta_sharpe_flat_haircut"]) > 0.0 and float(row["paired_ci_low_monthly"]) >= 0.0:
        return "watch"
    return "exclude"


def _write_report(
    path: Path,
    baseline_name: str,
    baseline: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 2 Exploration Summary",
        "",
        "Scope: exploration. These runs do not promote production specs or increment "
        "production trial count.",
        "",
        f"Baseline: `{baseline_name}` flat-haircut Sharpe "
        f"{baseline['sharpe_flat_haircut']:.3f}, turnover {baseline['turnover']:.1%}.",
        "",
        "| Candidate | Factors | Sharpe HC | Delta HC | Paired Delta Ann | "
        "Paired CI Monthly | Paired N | Turnover | Recommendation |",
        "|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['name']}` | {row['factors']} | {row['sharpe_flat_haircut']:.3f} | "
            f"{row['delta_sharpe_flat_haircut']:+.3f} | {row['paired_delta_annual']:+.1%} | "
            f"[{row['paired_ci_low_monthly']:+.2%}, {row['paired_ci_high_monthly']:+.2%}] | "
            f"{row['paired_n_obs']} | {row['turnover']:.1%} | {row['recommendation']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Shortlist review requires flat-haircut Sharpe at least +0.10 above "
        "`mom-s8`, positive paired-delta median, and paired CI lower bound >= 0.",
        "- Sector-dependent candidates are intentionally absent; all listed candidates "
        "use price/return-derived inputs only.",
        "- `watch` means directionally useful but below the formal incremental shortlist bar.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    candidates = sorted(args.candidate_dir.glob(args.pattern))
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    if not candidates:
        print(f"ERROR: no candidates matched {args.candidate_dir / args.pattern}", file=sys.stderr)
        return 1

    print(f"Loading baseline: {args.baseline}")
    baseline_spec = load_spec(args.baseline)
    baseline_result = run_from_spec(baseline_spec, raw_csv_path=args.raw_csv, write_ledger=False)
    baseline_wf = baseline_result.strategy_rolling
    baseline_row = {
        "sharpe_flat_haircut": baseline_wf.oos_sharpe_flat_haircut,
        "turnover": baseline_wf.oos_avg_turnover,
    }

    rows: list[dict[str, object]] = []
    for candidate_path in candidates:
        print(f"Running candidate: {candidate_path.name}")
        spec = load_spec(candidate_path)
        if spec.output.ledger_scope != "exploration":
            raise ValueError(f"{candidate_path} must remain exploration scope")
        result = run_from_spec(spec, raw_csv_path=args.raw_csv, write_ledger=False)
        wf = result.strategy_rolling
        paired = stationary_bootstrap_paired_delta(
            wf.oos_returns,
            baseline_wf.oos_returns,
            n_resamples=args.bootstrap_resamples,
            rng_seed=spec.master_seed,
        )
        row = {
            "name": spec.name,
            "factors": _factor_summary(spec),
            "sharpe_flat_haircut": wf.oos_sharpe_flat_haircut,
            "delta_sharpe_flat_haircut": (
                wf.oos_sharpe_flat_haircut - baseline_wf.oos_sharpe_flat_haircut
            ),
            "paired_delta_annual": paired.mean_delta_annual,
            "paired_ci_low_monthly": paired.ci_low_95_monthly,
            "paired_ci_median_monthly": paired.ci_median_monthly,
            "paired_ci_high_monthly": paired.ci_high_95_monthly,
            "paired_n_obs": paired.n_obs,
            "turnover": wf.oos_avg_turnover,
            "mean_annual": _mean_annual(wf.oos_returns),
        }
        row["incremental_bar"] = (
            row["delta_sharpe_flat_haircut"] >= 0.10
            and row["paired_ci_median_monthly"] > 0.0
            and row["paired_ci_low_monthly"] >= 0.0
        )
        row["recommendation"] = _recommend(row)
        rows.append(row)

    rows.sort(key=lambda row: float(row["delta_sharpe_flat_haircut"]), reverse=True)
    _write_report(args.out, baseline_spec.name, baseline_row, rows)
    print(f"Wrote exploration report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
