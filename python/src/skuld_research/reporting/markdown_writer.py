"""Deterministic markdown serialisation of MethodologyReport."""
from __future__ import annotations

from pathlib import Path

from skuld_common.contracts import MethodologyReport, WalkForwardResult


# Fixed column order for WalkForwardResult tables (determinism)
WF_TABLE_COLUMNS = (
    "oos_sharpe_raw",
    "oos_sharpe_flat_haircut",
    "oos_sharpe_delisting_adjusted",
    "oos_max_drawdown_observed",
    "oos_max_drawdown_augmented_median",
    "oos_max_drawdown_augmented_p90",
    "oos_avg_turnover",
    "oos_total_cost_nzd",
    "oos_hit_rate",
    "oos_skewness",
    "oos_calmar_ratio",
    "n_kept_folds",
    "n_rejected_folds",
)


def _format_wf_result_table(wf: WalkForwardResult, label: str) -> str:
    """Format WalkForwardResult as markdown table rows.
    
    Args:
        wf: WalkForwardResult to format.
        label: row label (e.g., "Strategy 2-fold").
    
    Returns:
        Markdown table rows (one row per result).
    """
    # Format each metric deterministically
    parts = [label]
    parts.append(f"{wf.oos_sharpe_raw:.3f}")
    parts.append(f"{wf.oos_sharpe_flat_haircut:.3f}")
    parts.append(f"{wf.oos_sharpe_delisting_adjusted:.3f}")
    parts.append(f"{wf.oos_max_drawdown_observed:.2%}")
    parts.append(f"{wf.oos_max_drawdown_augmented_median:.2%}")
    parts.append(f"{wf.oos_max_drawdown_augmented_p90:.2%}")
    parts.append(f"{wf.oos_avg_turnover:.2%}")
    parts.append(f"{wf.oos_total_cost_nzd:.0f}")
    parts.append(f"{wf.oos_hit_rate:.2%}")
    parts.append(f"{wf.oos_skewness:.3f}")
    parts.append(f"{wf.oos_calmar_ratio:.3f}")
    parts.append(f"{wf.n_kept_folds}")
    parts.append(f"{wf.n_rejected_folds}")
    
    return "| " + " | ".join(parts) + " |"


def write_methodology_report(report: MethodologyReport, path: Path) -> None:
    """Write MethodologyReport to deterministic markdown.
    
    DETERMINISTIC. No timestamps. Numeric formatting: percentages {:.2%},
    Sharpe {:.3f}, dates YYYY-MM-DD, no thousand separators in CIs.
    
    Args:
        report: MethodologyReport to serialize.
        path: output file path.
    """
    lines = []
    
    # Header
    lines.append("# Methodology Report")
    lines.append("")
    lines.append(f"**Config hash:** {report.config_hash}")
    lines.append(f"**Git SHA:** {report.git_sha}")
    lines.append(f"**As-of date:** {report.asof.strftime('%Y-%m-%d')}")
    lines.append(
        f"**Panel coverage:** {report.panel_coverage_start.strftime('%Y-%m-%d')} "
        f"to {report.panel_coverage_end.strftime('%Y-%m-%d')}"
    )
    lines.append(f"**Master seed:** {report.master_seed}")
    lines.append(f"**Prior trials:** {report.n_trials_prior}")
    lines.append("")
    lines.append("### RNG Master Seed Derivation")
    lines.append("")
    for line in report.rng_master_seed_note.strip().split("\n"):
        lines.append(line)
    lines.append("")
    
    # Strategy section
    lines.append("## Strategy")
    lines.append("")
    lines.append(f"**Name:** {report.strategy_name}")
    lines.append("")
    
    # 2-fold table
    lines.append("### Two-Fold Driver")
    lines.append("")
    lines.append(
        "| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | "
        "Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | "
        "Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |"
    )
    lines.append("|" + "---|" * 14)
    lines.append(_format_wf_result_table(report.strategy_two_fold, "Strategy"))
    lines.append("")
    
    # Rolling table (gating reference)
    lines.append("### Rolling Driver (Gating Reference)")
    lines.append("")
    lines.append(
        "| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | "
        "Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | "
        "Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |"
    )
    lines.append("|" + "---|" * 14)
    lines.append(_format_wf_result_table(report.strategy_rolling, "Strategy"))
    lines.append("")
    
    # Per-regime Sharpe
    if report.strategy_rolling.oos_sharpe_by_regime:
        lines.append("### Per-Regime Sharpe (Rolling Driver)")
        lines.append("")
        for regime, sharpe in sorted(report.strategy_rolling.oos_sharpe_by_regime.items()):
            lines.append(f"- **{regime}:** {sharpe:.3f}")
        lines.append("")
    
    # Benchmarks
    lines.append("## Benchmarks")
    lines.append("")
    
    for bench in report.benchmarks:
        lines.append(f"### {bench.name}")
        lines.append("")
        lines.append(
            f"**Coverage:** {bench.coverage_start.strftime('%Y-%m-%d')} "
            f"to {bench.coverage_end.strftime('%Y-%m-%d')}"
        )
        lines.append("")
        if bench.notes:
            lines.append("**Notes:**")
            for note in bench.notes:
                lines.append(f"- {note}")
            lines.append("")
        
        # Two-fold
        lines.append("#### Two-Fold Driver")
        lines.append("")
        lines.append(
            "| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | "
            "Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | "
            "Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |"
        )
        lines.append("|" + "---|" * 14)
        lines.append(_format_wf_result_table(bench.wf_two_fold, bench.name))
        lines.append("")
        
        # Rolling
        lines.append("#### Rolling Driver")
        lines.append("")
        lines.append(
            "| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | "
            "Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | "
            "Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |"
        )
        lines.append("|" + "---|" * 14)
        lines.append(_format_wf_result_table(bench.wf_rolling, bench.name))
        lines.append("")
    
    # Dominance
    lines.append("## Dominance (Romano-Wolf Stepwise)")
    lines.append("")
    lines.append("| Benchmark | Adjusted p-value | Dominates |")
    lines.append("|---|---|---|")
    
    for bench_name in report.dominance.benchmark_names:
        p_adj = report.dominance.adjusted_p_values[bench_name]
        dom = report.dominance.dominates[bench_name]
        lines.append(f"| {bench_name} | {p_adj:.4f} | {dom} |")
    lines.append("")
    
    # Gating decision
    lines.append("## Gating Decision")
    lines.append("")
    lines.append(f"**Overall:** {'PASS' if report.gating.passes else 'FAIL'}")
    lines.append("")
    lines.append("### Bars")
    lines.append("")
    for bar_name, (passed, reason) in report.gating.bars.items():
        status = "✓" if passed else "✗"
        lines.append(f"- **{bar_name}:** {status} — {reason}")
    lines.append("")
    
    lines.append(f"**Notes:** {report.gating.notes}")
    lines.append("")
    
    # Pass/Fail footer
    lines.append("## Pass / Fail")
    lines.append("")
    for bar_name, passed, reason in report.pass_fail:
        status = "✓ PASS" if passed else "✗ FAIL"
        lines.append(f"- **{bar_name}:** {status} — {reason}")
    lines.append("")
    
    # Rejection reasons (if any)
    if report.strategy_rolling.rejection_reasons:
        lines.append("### Rejected Folds")
        lines.append("")
        for reason in report.strategy_rolling.rejection_reasons:
            lines.append(f"- {reason}")
        lines.append("")
    
    # Write to file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
