"""WS1 attribution + WS3 investability filter comparison for mom-s8.

Usage (from python/ directory):
    uv run python scripts/ws1_ws3_analysis.py
    uv run python scripts/ws1_ws3_analysis.py --raw-csv ../data/data_long.csv

Outputs:
    reports/ws1_attribution.md   — full attribution + WS3 comparison table
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
from skuld_research.config.factors import build_factors_from_specs
from skuld_research.diagnostics.attribution import attribute_returns
from skuld_research.diagnostics.panels import score_panel
from skuld_research.factors.combiner import combine_signals
from skuld_research.stats.paired import stationary_bootstrap_paired_delta

_BASELINE = _PYTHON_ROOT / "configs" / "strategy-specs" / "production" / "mom-s8.yaml"
_WS3_SPECS = [
    _PYTHON_ROOT / "configs" / "strategy-specs" / "candidates" / f
    for f in [
        "ws3-mom-s8-adv25k.yaml",
        "ws3-mom-s8-hist180.yaml",
        "ws3-mom-s8-chronic3.yaml",
        "ws3-mom-s8-strict.yaml",
    ]
]
_DEFAULT_RAW_CSV = _PYTHON_ROOT.parent / "data" / "data_long.csv"
_DEFAULT_OUT = _PYTHON_ROOT / "reports" / "ws1_attribution.md"


# ---------------------------------------------------------------------------
# Score panel reconstruction
# ---------------------------------------------------------------------------

def _build_score_panels(
    result,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Reconstruct the combined + per-factor score panels from a RunResult.

    Returns:
        (combined_panel, component_panels) where:
          - combined_panel: date × ticker DataFrame of normalised combined z-scores
          - component_panels: {factor_name: date × ticker DataFrame of raw factor scores}
    """
    panel = result.panel
    spec = result.spec
    factors = build_factors_from_specs(spec.factors)
    all_tickers = sorted(panel.returns_monthly.columns.tolist())
    rebalance_dates = panel.universe_mask.index

    component_panels_rows: dict[str, list[pd.Series]] = {f.name: [] for f in factors}
    combined_rows: list[pd.Series] = []

    for t in rebalance_dates:
        mask = panel.universe_mask.loc[t]
        universe = mask[mask].index.tolist()

        if not universe:
            combined_rows.append(pd.Series(np.nan, index=all_tickers, dtype=float))
            for f in factors:
                component_panels_rows[f.name].append(
                    pd.Series(np.nan, index=all_tickers, dtype=float)
                )
            continue

        signals: dict[str, pd.Series] = {}
        for f in factors:
            signals[f.name] = f.score(panel, t, universe)

        combined = combine_signals(signals, universe, panel.sector, t)

        # Combined z-score (post-normalisation, used by engine)
        row_combined = combined.scores.reindex(all_tickers)
        combined_rows.append(row_combined)

        # Per-factor normalised component scores (cols from component_scores DataFrame)
        for f in factors:
            if f.name in combined.component_scores.columns:
                row_comp = combined.component_scores[f.name].reindex(all_tickers)
            else:
                row_comp = pd.Series(np.nan, index=all_tickers, dtype=float)
            component_panels_rows[f.name].append(row_comp)

    combined_panel = pd.DataFrame(
        combined_rows, index=rebalance_dates, columns=all_tickers
    )
    component_panels = {
        name: pd.DataFrame(rows, index=rebalance_dates, columns=all_tickers)
        for name, rows in component_panels_rows.items()
    }
    return combined_panel, component_panels


# ---------------------------------------------------------------------------
# Report helpers
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
    if v != v:  # nan
        return "n/a"
    return f"{v:+.{digits}%}"


def _fmt_f(v: float, digits: int = 3) -> str:
    if v != v:
        return "n/a"
    return f"{v:.{digits}f}"


def _top_contributors(ticker_contributions: pd.DataFrame, n: int = 5) -> list[tuple[str, float]]:
    """Return top-n tickers by mean contribution across periods they appeared."""
    if ticker_contributions.empty:
        return []
    mean_contrib = ticker_contributions.mean(axis=0).sort_values(ascending=False)
    return [(t, float(v)) for t, v in mean_contrib.head(n).items()]


def _bottom_contributors(ticker_contributions: pd.DataFrame, n: int = 5) -> list[tuple[str, float]]:
    if ticker_contributions.empty:
        return []
    mean_contrib = ticker_contributions.mean(axis=0).sort_values(ascending=True)
    return [(t, float(v)) for t, v in mean_contrib.head(n).items()]


def _cap_binding_stats(result) -> dict[str, float]:
    """Summary stats for cap_binding_count across all OOS folds."""
    counts = []
    for fold in result.strategy_rolling.folds:
        cbc = fold.result.cap_binding_count
        if not cbc.empty:
            counts.extend(cbc.tolist())
    if not counts:
        return {"mean": 0.0, "max": 0, "pct_nonzero": 0.0}
    arr = np.array(counts, dtype=float)
    return {
        "mean": float(arr.mean()),
        "max": int(arr.max()),
        "pct_nonzero": float((arr > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _write_report(
    path: Path,
    attribution,
    cap_stats: dict,
    baseline_wf,
    ws3_rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines += [
        "# WS1 Attribution + WS3 Investability Filter Analysis",
        "",
        "Scope: exploration. No production spec is modified.",
        "",
    ]

    # ------------------------------------------------------------------
    # WS1 attribution
    # ------------------------------------------------------------------
    lines += ["## WS1 — Return Attribution (mom-s8)", ""]

    # Headline metrics
    mkt_cagr = _cagr(attribution.market_proxy_monthly)
    sig_cagr = _cagr(attribution.signal_ew_monthly)
    prod_cagr = _cagr(attribution.production_monthly.dropna())

    lines += [
        "### Headline decomposition",
        "",
        f"| Component | Annualised CAGR |",
        f"|---|---:|",
        f"| Market proxy (EW universe) | {_fmt_pct(mkt_cagr)} |",
        f"| Signal EW (zero-cost top-50%) | {_fmt_pct(sig_cagr)} |",
        f"| Production (net of costs) | {_fmt_pct(prod_cagr)} |",
        f"| **Signal contribution (signal − mkt)** | **{_fmt_pct(attribution.signal_contribution_ann)}** |",
        f"| **Construction + cost drag (prod − signal)** | **{_fmt_pct(attribution.construction_cost_drag_ann)}** |",
        f"| **Total alpha (prod − mkt)** | **{_fmt_pct(attribution.total_alpha_ann)}** |",
        "",
    ]

    # Factor leg alphas
    if attribution.factor_leg_alpha_ann:
        lines += [
            "### Factor-leg standalone alpha vs market proxy",
            "",
            "| Factor | Standalone EW alpha (ann.) |",
            "|---|---:|",
        ]
        for fname, alpha in attribution.factor_leg_alpha_ann.items():
            lines.append(f"| {fname} | {_fmt_pct(alpha)} |")
        lines.append("")

    # Universe breadth
    breadth = attribution.breadth_series
    lines += [
        "### Universe breadth (tickers passing all filters per rebalance)",
        "",
        f"| Statistic | Value |",
        f"|---|---:|",
        f"| Mean | {breadth.mean():.1f} |",
        f"| Min | {int(breadth.min())} |",
        f"| Max | {int(breadth.max())} |",
        f"| Median | {breadth.median():.0f} |",
        "",
    ]

    # Cap binding
    lines += [
        "### Position cap binding (max_position=0.25)",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Mean bound tickers per period | {cap_stats['mean']:.2f} |",
        f"| Max bound tickers in one period | {cap_stats['max']} |",
        f"| % periods with any binding | {cap_stats['pct_nonzero']:.0%} |",
        "",
    ]

    # Top/bottom contributors
    top = _top_contributors(attribution.ticker_contributions)
    bot = _bottom_contributors(attribution.ticker_contributions)
    if top:
        lines += [
            "### Top-5 contributors to signal-EW (mean per-period weighted return)",
            "",
            "| Ticker | Mean contribution |",
            "|---|---:|",
        ]
        for t, v in top:
            lines.append(f"| {t} | {v:+.4f} |")
        lines.append("")
    if bot:
        lines += [
            "### Bottom-5 contributors to signal-EW",
            "",
            "| Ticker | Mean contribution |",
            "|---|---:|",
        ]
        for t, v in bot:
            lines.append(f"| {t} | {v:+.4f} |")
        lines.append("")

    # OOS regime Sharpe
    regime_sharpe = getattr(baseline_wf, "oos_sharpe_by_regime", {})
    if regime_sharpe:
        lines += [
            "### OOS Sharpe by regime",
            "",
            "| Regime | Sharpe |",
            "|---|---:|",
        ]
        for regime, sh in regime_sharpe.items():
            lines.append(f"| {regime} | {sh:.3f} |")
        lines.append("")

    # ------------------------------------------------------------------
    # WS3 comparison
    # ------------------------------------------------------------------
    if ws3_rows:
        baseline_sh = baseline_wf.oos_sharpe_flat_haircut
        baseline_to = baseline_wf.oos_avg_turnover
        baseline_n_obs = len(attribution.production_monthly.dropna())

        lines += [
            "## WS3 — Investability Filter Variants",
            "",
            "Decision criterion: universe breadth must stay >= 6 names on average "
            "and flat-haircut Sharpe must not worsen by more than 0.05 vs baseline.",
            "",
            f"Baseline `mom-s8`: flat-haircut Sharpe {baseline_sh:.3f}, "
            f"turnover {baseline_to:.1%}, OOS n={baseline_n_obs}.",
            "",
            "| Variant | Universe breadth (mean) | Sharpe HC | Delta HC | "
            "Paired delta ann. | Paired CI monthly | Turnover | Assessment |",
            "|---|---:|---:|---:|---:|---|---:|---|",
        ]
        for row in ws3_rows:
            verdict = "ok" if row["viable"] else "too-thin"
            if row["viable"]:
                verdict = "pass" if row["delta_sharpe"] >= -0.05 else "worse"
            lines.append(
                f"| `{row['name']}` "
                f"| {row['breadth_mean']:.1f} "
                f"| {row['sharpe_hc']:.3f} "
                f"| {row['delta_sharpe']:+.3f} "
                f"| {row['paired_delta_annual']:+.1%} "
                f"| [{row['paired_ci_low']:+.2%}, {row['paired_ci_high']:+.2%}] "
                f"| {row['turnover']:.1%} "
                f"| {verdict} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote report: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WS1 attribution + WS3 investability analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-csv", type=Path, default=_DEFAULT_RAW_CSV)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--bootstrap-resamples", type=int, default=500)
    p.add_argument("--skip-ws3", action="store_true", help="Skip WS3 variant runs")
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
    print("Step 1/3: Running mom-s8 baseline …")
    print("=" * 60)
    baseline_spec = load_spec(_BASELINE)
    baseline_result = run_from_spec(
        baseline_spec, raw_csv_path=args.raw_csv, write_ledger=False
    )
    baseline_wf = baseline_result.strategy_rolling

    print(
        f"  OOS Sharpe (raw): {baseline_wf.oos_sharpe_raw:.3f} | "
        f"flat-haircut: {baseline_wf.oos_sharpe_flat_haircut:.3f} | "
        f"turnover: {baseline_wf.oos_avg_turnover:.1%} | "
        f"folds kept: {baseline_wf.n_kept_folds}/{baseline_wf.n_kept_folds + baseline_wf.n_rejected_folds}"
    )

    # ------------------------------------------------------------------
    # Step 2: Compute WS1 attribution
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2/3: Computing WS1 attribution …")
    print("=" * 60)

    print("  Building combined + component score panels …")
    combined_panel, component_panels = _build_score_panels(baseline_result)
    factor_names = list(component_panels.keys())
    print(f"  Factors: {factor_names} | rebalance dates: {len(combined_panel)}")

    oos_returns = baseline_wf.oos_returns
    attribution = attribute_returns(
        combined_panel,
        baseline_result.panel,
        oos_returns,
        top_frac=0.5,
        component_score_panels=component_panels,
    )
    cap_stats = _cap_binding_stats(baseline_result)

    print(
        f"  Signal contribution: {attribution.signal_contribution_ann:+.1%} ann. | "
        f"Construction+cost drag: {attribution.construction_cost_drag_ann:+.1%} ann. | "
        f"Total alpha: {attribution.total_alpha_ann:+.1%} ann."
    )
    print(f"  Universe breadth: mean={attribution.breadth_series.mean():.1f} "
          f"min={int(attribution.breadth_series.min())} max={int(attribution.breadth_series.max())}")
    print(f"  Factor leg alphas: "
          + ", ".join(f"{k}={v:+.1%}" for k, v in attribution.factor_leg_alpha_ann.items()))
    print(f"  Cap binding: mean={cap_stats['mean']:.2f} max={cap_stats['max']} "
          f"nonzero={cap_stats['pct_nonzero']:.0%} of periods")

    # ------------------------------------------------------------------
    # Step 3: WS3 investability variants
    # ------------------------------------------------------------------
    ws3_rows: list[dict] = []
    if not args.skip_ws3:
        print("\n" + "=" * 60)
        print("Step 3/3: Running WS3 investability variants …")
        print("=" * 60)

        for spec_path in _WS3_SPECS:
            if not spec_path.exists():
                print(f"  SKIP (not found): {spec_path.name}")
                continue
            print(f"\n  Running {spec_path.name} …")
            spec = load_spec(spec_path)
            try:
                result = run_from_spec(
                    spec, raw_csv_path=args.raw_csv, write_ledger=False
                )
            except Exception as exc:
                print(f"    ERROR: {exc}")
                continue

            wf = result.strategy_rolling
            paired = stationary_bootstrap_paired_delta(
                wf.oos_returns,
                baseline_wf.oos_returns,
                n_resamples=args.bootstrap_resamples,
                rng_seed=spec.master_seed,
            )

            # Universe breadth from this variant's panel
            breadth_mean = float(result.panel.universe_mask.sum(axis=1).mean())
            viable = breadth_mean >= 6.0

            row = {
                "name": spec.name,
                "breadth_mean": breadth_mean,
                "sharpe_hc": wf.oos_sharpe_flat_haircut,
                "delta_sharpe": wf.oos_sharpe_flat_haircut - baseline_wf.oos_sharpe_flat_haircut,
                "paired_delta_annual": paired.mean_delta_annual,
                "paired_ci_low": paired.ci_low_95_monthly,
                "paired_ci_high": paired.ci_high_95_monthly,
                "turnover": wf.oos_avg_turnover,
                "viable": viable,
                "n_obs": len(wf.oos_returns.dropna()),
            }
            ws3_rows.append(row)
            print(
                f"    Sharpe HC: {row['sharpe_hc']:.3f} ({row['delta_sharpe']:+.3f}) | "
                f"breadth: {breadth_mean:.1f} | turnover: {row['turnover']:.1%} | "
                f"viable: {viable}"
            )

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    _write_report(
        args.out,
        attribution,
        cap_stats,
        baseline_wf,
        ws3_rows,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
