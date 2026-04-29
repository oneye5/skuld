"""Information coefficient computation for factor signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from skuld_common.contracts import ICReport


def ranking_ic(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    horizon_months: int,
    *,
    factor_name: str = "factor",
    min_cross_section: int = 10,
) -> ICReport:
    """Compute IC between factor scores and forward returns.

    Args:
        factor_panel: index=rebalance_date (month-end), columns=ticker,
            values=factor score at t.
        returns_panel: index=month-end, columns=ticker, values=monthly total return.
        horizon_months: forward return horizon (>=1).
        factor_name: name of the factor for the report.
        min_cross_section: minimum number of valid ticker observations per date
            to compute IC; dates with fewer are dropped.

    Returns:
        ICReport with IC time series and summary statistics.
    """
    ic_series_data = []
    min_universe = float("inf")

    for t in factor_panel.index:
        # Get factor scores at t
        scores = factor_panel.loc[t].dropna()

        # Compute forward return over (t, t+horizon_months]
        # We need to find returns from t+1 to t+horizon_months
        future_dates = returns_panel.index[returns_panel.index > t]
        if len(future_dates) < horizon_months:
            # Not enough future data
            continue

        # Get the next horizon_months returns
        end_date = future_dates[horizon_months - 1]
        fwd_returns = (
            (1.0 + returns_panel.loc[future_dates[:horizon_months]])
            .prod()
            - 1.0
        )

        # Align scores and forward returns
        common_tickers = scores.index.intersection(fwd_returns.index)
        if len(common_tickers) < min_cross_section:
            continue

        scores_aligned = scores[common_tickers]
        fwd_aligned = fwd_returns[common_tickers]

        # Drop any remaining NaNs
        valid = scores_aligned.notna() & fwd_aligned.notna()
        if valid.sum() < min_cross_section:
            continue

        scores_clean = scores_aligned[valid]
        fwd_clean = fwd_aligned[valid]

        # Compute Spearman rank correlation
        if len(scores_clean) >= min_cross_section:
            ic_val, _ = spearmanr(scores_clean, fwd_clean)
            ic_series_data.append((t, ic_val))
            min_universe = min(min_universe, len(scores_clean))

    if not ic_series_data:
        # No valid observations
        ic_series = pd.Series([], dtype=float)
        return ICReport(
            factor_name=factor_name,
            horizon_months=horizon_months,
            ic_series=ic_series,
            ic_mean=float("nan"),
            ic_std=float("nan"),
            ic_ir=float("nan"),
            t_stat_newey_west=float("nan"),
            n_obs=0,
            min_universe_per_date=0,
        )

    ic_series = pd.Series(
        [ic for _, ic in ic_series_data],
        index=[t for t, _ in ic_series_data],
    )

    # Compute summary statistics
    ic_mean = ic_series.mean()
    ic_std = ic_series.std(ddof=1)
    n_obs = len(ic_series)

    # Annualized IC IR (Grinold/Kahn convention)
    if ic_std > 1e-10:  # Avoid division by zero for perfect correlations
        ic_ir = ic_mean / ic_std * np.sqrt(12.0 / horizon_months)
    else:
        ic_ir = float("nan")

    # Newey-West t-statistic
    t_stat_nw = _newey_west_t_stat(ic_series, lag=horizon_months)

    return ICReport(
        factor_name=factor_name,
        horizon_months=horizon_months,
        ic_series=ic_series,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_ir=ic_ir,
        t_stat_newey_west=t_stat_nw,
        n_obs=n_obs,
        min_universe_per_date=int(min_universe) if min_universe != float("inf") else 0,
    )


def _newey_west_t_stat(series: pd.Series, lag: int) -> float:
    """Compute Newey-West t-statistic for the mean of a time series.

    Args:
        series: time series of observations.
        lag: maximum lag for HAC variance estimator.

    Returns:
        t-statistic using Newey-West standard error.
    """
    if len(series) == 0:
        return float("nan")

    mean = series.mean()
    demeaned = series - mean
    n = len(series)

    # Variance estimator: γ_0 + 2 * Σ_{k=1}^{L} (1 - k/(L+1)) * γ_k
    gamma_0 = (demeaned**2).mean()
    variance_nw = gamma_0

    for k in range(1, min(lag + 1, n)):
        weight = 1.0 - k / (lag + 1)
        gamma_k = (demeaned.iloc[:-k] * demeaned.iloc[k:].values).mean()
        variance_nw += 2.0 * weight * gamma_k

    if variance_nw <= 0:
        return float("nan")

    se_nw = np.sqrt(variance_nw / n)
    if se_nw == 0:
        return float("nan")

    return mean / se_nw
