"""Market regime labelling based on trailing 12-month returns."""
from __future__ import annotations

import pandas as pd

from skuld_common.contracts import PreparedPanel


def label_regimes(panel: PreparedPanel) -> pd.Series:
    """Label each month with market regime based on trailing-12m proxy return.
    
    Uses equal-weighted market proxy (liquid universe) as stand-in for NZX50.
    For each month t, computes trailing-12m return using months strictly < t (PIT).
    
    Regime rules:
    - trailing_12m > +0.10 → "bull"
    - trailing_12m < -0.10 → "bear"
    - otherwise → "chop"
    - first 12 months → "chop" (insufficient history)
    
    Args:
        panel: PreparedPanel with returns_monthly and universe_mask.
    
    Returns:
        Series indexed by panel.returns_monthly.index with regime labels.
    """
    # Build market proxy: equal-weighted across liquid universe
    ever_in_universe = panel.universe_mask.any(axis=0)
    liquid_tickers = ever_in_universe[ever_in_universe].index.tolist()

    if not liquid_tickers:
        # No liquid universe → all chop
        return pd.Series("chop", index=panel.returns_monthly.index)

    market_ret = panel.returns_monthly[liquid_tickers].mean(axis=1)

    labels = []
    dates = market_ret.index.tolist()

    for i, t in enumerate(dates):
        if i < 12:
            # Insufficient history
            labels.append("chop")
        else:
            # Trailing 12 months: strictly before t
            trailing_12 = market_ret.iloc[i - 12 : i]
            cumulative_ret = (1.0 + trailing_12).prod() - 1.0

            if cumulative_ret > 0.10:
                labels.append("bull")
            elif cumulative_ret < -0.10:
                labels.append("bear")
            else:
                labels.append("chop")

    return pd.Series(labels, index=dates, name="regime")
