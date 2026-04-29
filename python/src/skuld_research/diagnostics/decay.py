"""Alpha decay analysis across forward horizons."""

from __future__ import annotations

import pandas as pd

from skuld_common.contracts import DecayReport, ICReport
from skuld_research.diagnostics.ic import ranking_ic


def alpha_decay(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 3, 6, 12),
    *,
    factor_name: str = "factor",
    min_cross_section: int = 10,
) -> DecayReport:
    """Compute IC across multiple forward horizons.

    Args:
        factor_panel: index=rebalance_date, columns=ticker, values=factor score.
        returns_panel: index=month-end, columns=ticker, values=monthly return.
        horizons: tuple of forward horizons in months.
        factor_name: name of the factor.
        min_cross_section: minimum cross-sectional sample per IC observation.

    Returns:
        DecayReport with IC at each horizon and peak horizon.
    """
    ic_by_horizon = {}
    for h in horizons:
        ic_report = ranking_ic(
            factor_panel,
            returns_panel,
            horizon_months=h,
            factor_name=factor_name,
            min_cross_section=min_cross_section,
        )
        ic_by_horizon[h] = ic_report

    # Find peak horizon (argmax IC mean)
    peak_horizon = max(horizons, key=lambda h: ic_by_horizon[h].ic_mean)

    return DecayReport(
        factor_name=factor_name,
        horizons=horizons,
        ic_by_horizon=ic_by_horizon,
        peak_horizon=peak_horizon,
    )
