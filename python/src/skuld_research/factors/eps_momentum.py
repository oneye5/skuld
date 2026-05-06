"""EPS momentum factor.

Year-over-year growth of trailing diluted EPS, using only data published
strictly before the rebalance date (PIT-safe).  Higher growth = more positive
signal; earnings acceleration is associated with subsequent outperformance.

When the base-year EPS is near zero the growth ratio becomes unstable.  We
cap the raw ratio at ±10× before cross-sectional standardisation so a single
extreme re-rating does not dominate the universe ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PreparedPanel

_EPS_FIELD = "trailing_diluted_eps"
_LOOKBACK_MONTHS = 12
_MAX_ABS_GROWTH = 10.0


class EpsMomentumFactor:
    """Cross-sectional YoY EPS growth signal."""

    name: str = "eps_momentum"

    def score(
        self,
        panel: PreparedPanel,
        t: pd.Timestamp,
        universe: list[str],
    ) -> pd.Series:
        fundamentals = panel.fundamentals
        t_naive = t.tz_localize(None) if t.tzinfo else t
        nan_series = pd.Series(np.nan, index=universe, dtype=float, name=self.name)

        if fundamentals.empty or _EPS_FIELD not in fundamentals.columns:
            return nan_series

        eps_series = fundamentals[_EPS_FIELD].dropna()
        if eps_series.empty:
            return nan_series

        cutoff_now = t_naive
        cutoff_year_ago = t_naive - pd.DateOffset(months=_LOOKBACK_MONTHS)

        scores: dict[str, float] = {}
        for ticker in universe:
            try:
                ticker_eps = eps_series.xs(ticker, level="ticker")
            except KeyError:
                scores[ticker] = np.nan
                continue

            ticker_eps = ticker_eps.sort_index()
            # PIT: only publications strictly before t
            past = ticker_eps[ticker_eps.index < cutoff_now]
            if past.empty:
                scores[ticker] = np.nan
                continue

            eps_now = float(past.iloc[-1])

            # Latest EPS published before t - 12 months
            year_ago = past[past.index < cutoff_year_ago]
            if year_ago.empty:
                scores[ticker] = np.nan
                continue

            eps_base = float(year_ago.iloc[-1])

            # Growth ratio: undefined when base ≈ 0; clamp at ±_MAX_ABS_GROWTH
            denom = abs(eps_base)
            if denom < 1e-9:
                scores[ticker] = np.nan
                continue

            growth = (eps_now - eps_base) / denom
            scores[ticker] = float(np.clip(growth, -_MAX_ABS_GROWTH, _MAX_ABS_GROWTH))

        return pd.Series(scores, dtype=float, name=self.name).reindex(universe)
