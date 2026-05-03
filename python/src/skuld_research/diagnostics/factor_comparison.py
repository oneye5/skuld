"""Factor IC comparison: standalone IC/decay per factor + pairwise IC-series correlation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from skuld_common.contracts import DecayReport, ICReport, PreparedPanel
from skuld_research.diagnostics.decay import alpha_decay
from skuld_research.diagnostics.ic import ranking_ic
from skuld_research.diagnostics.panels import score_panel
from skuld_research.factors.protocols import SignalGenerator


@dataclass(frozen=True)
class FactorComparisonReport:
    """Comparison of multiple factors by IC, decay, and redundancy."""

    factor_names: tuple[str, ...]
    ic_reports: dict[str, ICReport]
    decay_reports: dict[str, DecayReport]
    ic_series_corr: pd.DataFrame
    redundant_pairs: tuple[tuple[str, str], ...]
    redundancy_threshold: float

    __hash__ = None  # pd.DataFrame field is unhashable


def compare_factors(
    factors: dict[str, SignalGenerator],
    panel: PreparedPanel,
    *,
    horizon_months: int = 1,
    decay_horizons: tuple[int, ...] = (1, 2, 3, 6, 12),
    min_cross_section: int = 3,
    redundancy_threshold: float = 0.5,
) -> FactorComparisonReport:
    """Compare multiple factors by IC, decay curve, and pairwise IC-series correlation."""
    factor_names = tuple(factors.keys())

    if not factor_names:
        empty_corr = pd.DataFrame(dtype=float)
        return FactorComparisonReport(
            factor_names=(),
            ic_reports={},
            decay_reports={},
            ic_series_corr=empty_corr,
            redundant_pairs=(),
            redundancy_threshold=redundancy_threshold,
        )

    ic_reports: dict[str, ICReport] = {}
    decay_reports: dict[str, DecayReport] = {}

    for name, factor in factors.items():
        fp = score_panel(factor, panel)
        ic_reports[name] = ranking_ic(
            fp,
            panel.returns_monthly,
            horizon_months,
            factor_name=name,
            min_cross_section=min_cross_section,
        )
        decay_reports[name] = alpha_decay(
            fp,
            panel.returns_monthly,
            decay_horizons,
            factor_name=name,
            min_cross_section=min_cross_section,
        )

    # Build IC series correlation matrix
    ic_series: dict[str, pd.Series] = {
        name: ic_reports[name].ic_series for name in factor_names
    }

    corr_data = np.full((len(factor_names), len(factor_names)), np.nan)
    for i, a in enumerate(factor_names):
        corr_data[i, i] = 1.0
        for j, b in enumerate(factor_names):
            if j <= i:
                continue
            common = ic_series[a].index.intersection(ic_series[b].index)
            sa = ic_series[a].reindex(common).dropna()
            sb = ic_series[b].reindex(common).dropna()
            common2 = sa.index.intersection(sb.index)
            if len(common2) < 3:
                continue
            r, _ = stats.spearmanr(sa.reindex(common2), sb.reindex(common2))
            corr_data[i, j] = r
            corr_data[j, i] = r

    ic_series_corr = pd.DataFrame(
        corr_data, index=list(factor_names), columns=list(factor_names)
    )

    # Find redundant pairs
    redundant_pairs: list[tuple[str, str]] = []
    for i, a in enumerate(factor_names):
        for j, b in enumerate(factor_names):
            if j <= i:
                continue
            val = ic_series_corr.loc[a, b]
            if not np.isnan(val) and abs(val) > redundancy_threshold:
                redundant_pairs.append((a, b))

    return FactorComparisonReport(
        factor_names=factor_names,
        ic_reports=ic_reports,
        decay_reports=decay_reports,
        ic_series_corr=ic_series_corr,
        redundant_pairs=tuple(redundant_pairs),
        redundancy_threshold=redundancy_threshold,
    )
