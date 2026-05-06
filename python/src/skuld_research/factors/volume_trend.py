"""Volume trend factor.

Ratio of recent average daily volume to a longer-term baseline, computed
strictly from data available before the rebalance date (PIT-safe).

Score = log(ADV_short / ADV_long), where:
  - ADV_short = mean daily volume over the trailing ``short_days`` calendar
    days (default 20, approximately one month of trading).
  - ADV_long  = mean daily volume over the trailing ``long_days`` calendar
    days (default 60, approximately one quarter of trading).

A positive score means volume is accelerating: more recent activity is above
the quarterly baseline.  Accelerating volume alongside positive price momentum
is associated with conviction and continuation.

Non-trading days carry NaN volume; we use ``dropna()`` to work only on actual
trading days, preserving the ratio's meaning even for thinly-traded names.
Tickers with fewer than ``min_trading_days`` non-NaN observations in the long
window are excluded (return NaN) to avoid noisy ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PreparedPanel

_SHORT_DAYS = 20
_LONG_DAYS = 60
_MIN_TRADING_DAYS = 30


class VolumeTrendFactor:
    """Cross-sectional volume acceleration signal."""

    name: str = "volume_trend"

    def __init__(
        self,
        short_days: int = _SHORT_DAYS,
        long_days: int = _LONG_DAYS,
        min_trading_days: int = _MIN_TRADING_DAYS,
    ) -> None:
        if short_days >= long_days:
            raise ValueError(f"short_days ({short_days}) must be < long_days ({long_days})")
        self.short_days = short_days
        self.long_days = long_days
        self.min_trading_days = min_trading_days

    def score(
        self,
        panel: PreparedPanel,
        t: pd.Timestamp,
        universe: list[str],
    ) -> pd.Series:
        volumes = panel.volumes
        t_naive = t.tz_localize(None) if t.tzinfo else t
        nan_series = pd.Series(np.nan, index=universe, dtype=float, name=self.name)

        if volumes.empty:
            return nan_series

        # Restrict to dates strictly before t (PIT)
        avail = volumes[volumes.index < t_naive]
        if avail.empty:
            return nan_series

        long_window = avail.iloc[-self.long_days :]

        scores: dict[str, float] = {}
        for ticker in universe:
            if ticker not in avail.columns:
                scores[ticker] = np.nan
                continue

            long_series = long_window[ticker].dropna()
            if len(long_series) < self.min_trading_days:
                scores[ticker] = np.nan
                continue

            short_series = long_series.iloc[-self.short_days :]
            if short_series.empty:
                scores[ticker] = np.nan
                continue

            adv_long = float(long_series.mean())
            adv_short = float(short_series.mean())

            if adv_long <= 0:
                scores[ticker] = np.nan
                continue

            # log-ratio: symmetric, naturally scaled
            scores[ticker] = float(np.log(adv_short / adv_long + 1e-12))

        return pd.Series(scores, dtype=float, name=self.name).reindex(universe)
