"""One-sided HAC excess-return tests."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from skuld_common.contracts import ExcessReturnTestResult


def _newey_west_t_stat(series: pd.Series, lag: int) -> float:
    """Compute a Newey-West t-statistic for the sample mean."""
    if len(series) == 0:
        return float("nan")

    mean = float(series.mean())
    demeaned = series - mean
    n = len(series)
    gamma_0 = float((demeaned**2).mean())
    variance_nw = gamma_0

    for k in range(1, min(lag + 1, n)):
        weight = 1.0 - k / (lag + 1)
        gamma_k = float((demeaned.iloc[:-k] * demeaned.iloc[k:].values).mean())
        variance_nw += 2.0 * weight * gamma_k

    if variance_nw <= 0:
        if mean > 0.0:
            return float("inf")
        if mean < 0.0:
            return float("-inf")
        return float("nan")

    se_nw = math.sqrt(variance_nw / n)
    if se_nw == 0.0:
        return float("nan")

    return mean / se_nw


def one_sided_hac_excess_return(
    strategy: pd.Series,
    benchmark: pd.Series,
    alpha: float = 0.05,
) -> ExcessReturnTestResult:
    """Test whether strategy mean return exceeds benchmark mean return."""
    aligned = pd.concat([strategy, benchmark], axis=1, join="inner").dropna()
    if aligned.empty:
        return ExcessReturnTestResult(
            mean_excess_annual=float("nan"),
            t_stat=float("nan"),
            p_value=float("nan"),
            passes=False,
        )

    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    lag = max(1, int(np.floor(len(excess) ** 0.25)))
    t_stat = _newey_west_t_stat(excess, lag=lag)
    if t_stat == float("inf"):
        p_value = 0.0
    elif t_stat == float("-inf"):
        p_value = 1.0
    elif math.isfinite(t_stat):
        p_value = float(scipy_stats.norm.sf(t_stat))
    else:
        p_value = float("nan")
    mean_excess_annual = float(excess.mean()) * 12.0
    passes = mean_excess_annual > 0.0 and math.isfinite(p_value) and p_value <= alpha

    return ExcessReturnTestResult(
        mean_excess_annual=mean_excess_annual,
        t_stat=t_stat,
        p_value=p_value,
        passes=passes,
    )
