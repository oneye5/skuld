"""Helper functions for building factor score panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PreparedPanel
from skuld_research.factors.protocols import SignalGenerator


def score_panel(factor: SignalGenerator, panel: PreparedPanel) -> pd.DataFrame:
    """Build a wide factor-score panel across all rebalance dates.

    Args:
        factor: SignalGenerator to score.
        panel: PreparedPanel with universe_mask and returns.

    Returns:
        DataFrame indexed by rebalance_date, columns=ticker, values=factor score.
    """
    rebalance_dates = panel.universe_mask.index
    all_tickers = sorted(panel.returns_monthly.columns.tolist())

    scores_by_date = []

    for t in rebalance_dates:
        # Get universe for this date
        universe_mask = panel.universe_mask.loc[t]
        universe = universe_mask[universe_mask].index.tolist()

        if not universe:
            # No valid universe at this date
            scores = pd.Series(np.nan, index=all_tickers, dtype=float)
        else:
            # Score the universe
            scores = factor.score(panel, t, universe)
            # Reindex to all tickers
            scores = scores.reindex(all_tickers)

        scores_by_date.append(scores)

    return pd.DataFrame(scores_by_date, index=rebalance_dates, columns=all_tickers)


def quintile_spread_returns(
    score_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.Series:
    """Compute long-short top-quintile minus bottom-quintile returns.

    Args:
        score_panel: index=rebalance_date, columns=ticker, values=scores.
        returns_panel: index=month-end, columns=ticker, values=monthly returns.
        n_quantiles: number of quantiles (default 5 for quintiles).

    Returns:
        Monthly Series of long-short returns.
    """
    spreads = []

    for t in score_panel.index:
        scores = score_panel.loc[t].dropna()

        if len(scores) < n_quantiles:
            # Not enough tickers to form quantiles
            spreads.append((t, np.nan))
            continue

        # Compute quantiles
        quantiles = pd.qcut(scores, q=n_quantiles, labels=False, duplicates="drop")

        if quantiles.nunique() < 2:
            # All scores are the same or not enough variation
            spreads.append((t, np.nan))
            continue

        # Top quantile (highest scores)
        top_q = n_quantiles - 1
        top_tickers = scores[quantiles == top_q].index.tolist()

        # Bottom quantile (lowest scores)
        bottom_q = 0
        bottom_tickers = scores[quantiles == bottom_q].index.tolist()

        # Get next month's return (t+1)
        future_dates = returns_panel.index[returns_panel.index > t]
        if len(future_dates) == 0:
            spreads.append((t, np.nan))
            continue

        next_month = future_dates[0]

        # Equal-weighted returns for top and bottom
        top_ret = returns_panel.loc[next_month, top_tickers].mean()
        bottom_ret = returns_panel.loc[next_month, bottom_tickers].mean()

        spread = top_ret - bottom_ret
        # Use the next_month as the index (this is the return period)
        spreads.append((next_month, spread))

    if not spreads:
        return pd.Series([], dtype=float)

    # Build Series, handling potential duplicate indices
    spread_dict = {}
    for date, val in spreads:
        # If we have duplicate months, keep the first one
        if date not in spread_dict:
            spread_dict[date] = val

    return pd.Series(spread_dict).sort_index()


def market_proxy_returns(panel: PreparedPanel) -> pd.Series:
    """Equal-weighted market proxy from the liquid universe.

    Args:
        panel: PreparedPanel with returns_monthly and universe_mask.

    Returns:
        Monthly Series of equal-weighted market returns.
    """
    # Liquid universe: any ticker that ever passed the universe filter
    ever_in_universe = panel.universe_mask.any(axis=0)
    liquid_tickers = ever_in_universe[ever_in_universe].index.tolist()

    if not liquid_tickers:
        return pd.Series([], dtype=float)

    # Equal-weighted mean across liquid tickers, ignoring NaNs
    market_ret = panel.returns_monthly[liquid_tickers].mean(axis=1)

    return market_ret
