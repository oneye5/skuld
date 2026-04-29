"""Markdown report writer for diagnostics."""

from __future__ import annotations

from pathlib import Path

from skuld_common.contracts import DecayReport, DecompositionReport, ICReport


def write_diagnostics_report(
    ic: ICReport,
    decay: DecayReport,
    decomp: DecompositionReport,
    out_path: Path,
) -> None:
    """Write diagnostics markdown report.

    Args:
        ic: IC report for primary horizon.
        decay: Decay report across multiple horizons.
        decomp: Decomposition report.
        out_path: Output file path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Signal Diagnostics: {ic.factor_name}")
    lines.append("")

    # IC Summary Section
    lines.append("## Information Coefficient (IC)")
    lines.append("")
    lines.append(f"**Horizon:** {ic.horizon_months} month(s)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| IC Mean | {ic.ic_mean:.4f} |")
    lines.append(f"| IC Std | {ic.ic_std:.4f} |")
    lines.append(f"| IC IR (Annualized) | {ic.ic_ir:.4f} |")
    lines.append(f"| Newey-West t-stat | {ic.t_stat_newey_west:.4f} |")
    lines.append(f"| Observations | {ic.n_obs} |")
    lines.append(f"| Min Universe Size | {ic.min_universe_per_date} |")
    lines.append("")

    # Decay Section
    lines.append("## Alpha Decay Across Horizons")
    lines.append("")
    lines.append("| Horizon (months) | IC Mean | IC Std | IC IR | t-stat | N |")
    lines.append("|------------------|---------|--------|-------|--------|---|")

    for h in sorted(decay.horizons):
        ic_h = decay.ic_by_horizon[h]
        lines.append(
            f"| {h} | {ic_h.ic_mean:.4f} | {ic_h.ic_std:.4f} | "
            f"{ic_h.ic_ir:.4f} | {ic_h.t_stat_newey_west:.4f} | {ic_h.n_obs} |"
        )

    lines.append("")
    lines.append(f"**Peak Horizon:** {decay.peak_horizon} month(s)")
    lines.append("")

    # Decomposition Section
    lines.append("## Factor Decomposition")
    lines.append("")
    lines.append("| Regressor | Beta | t-stat |")
    lines.append("|-----------|------|--------|")

    for reg in decomp.regressors:
        beta = decomp.coefficients[reg]
        t_stat = decomp.t_stats[reg]
        lines.append(f"| {reg} | {beta:.4f} | {t_stat:.4f} |")

    lines.append("")
    lines.append("**Residual Alpha (Annualized):** "
                 f"{decomp.residual_alpha_annualised:.4f} "
                 f"(t={decomp.residual_alpha_t_stat:.4f})")
    lines.append("")
    lines.append(f"**R²:** {decomp.r_squared:.4f}")
    lines.append("")
    lines.append(f"**Observations:** {decomp.n_obs}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
