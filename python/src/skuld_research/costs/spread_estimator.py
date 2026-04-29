"""Abdi-Ranaldo (2017) effective bid-ask spread estimator from daily OHLC.

Reference:
    Abdi, F., & Ranaldo, A. (2017). "A Simple Estimation of Bid-Ask Spreads
    from Daily Close, High, and Low Prices." Review of Financial Studies,
    30(12), 4437-4480.

Per-day spread proxy:

    s_t = 2 * sqrt( max( (c_t - eta_t) * (c_t - eta_{t+1}), 0 ) )

where c_t = log(close_t) and eta_t = (log(high_t) + log(low_t)) / 2.

The result is an *effective* spread (what you actually pay, including
typical slippage), not a quoted spread. It is dimensionless (a fraction of
mid price) and is converted to per-side bps by:

    bps_per_side = (s / 2) * 1e4

since the effective spread `s` represents the round-trip cost as a fraction
of price; one side is half of that.

The estimator is averaged over a trailing window (default 60 trading days)
to produce a stable per-ticker estimate. It is point-in-time clean by
construction: at any rebalance date, only OHLC observations strictly before
that date are used.

Caveats:
    * The squared term inside the sqrt can be negative in low-volatility
      periods; we clip at 0 (per the original paper).
    * If a ticker has fewer than `min_obs` valid daily estimates in the
      window, the result is NaN; callers should fall back to a configurable
      default in that case.
    * Use raw (unadjusted) OHLC. Adjusted prices distort the H-L range
      retroactively when splits/dividends occur, which biases the estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Floor on per-side bps when the estimator returns NaN/zero. A floor is
# necessary because real markets always have *some* friction and we don't
# want degenerate zero-cost trades to leak into the backtest.
DEFAULT_MIN_BPS_PER_SIDE = 5.0


def compute_abdi_ranaldo_spread_panel(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    *,
    window: int = 60,
    min_obs: int = 20,
    scale: float = 1.0,
    min_bps_per_side: float = DEFAULT_MIN_BPS_PER_SIDE,
) -> pd.DataFrame:
    """Estimate per-side bid-ask spread (bps) per ticker per date.

    Args:
        high: date x ticker, raw high prices.
        low: date x ticker, raw low prices.
        close: date x ticker, raw close prices.
        window: trailing window length (trading days) for averaging the
            per-day estimator. Default 60.
        min_obs: minimum number of valid daily estimates required in the
            window for the rolling mean to be returned. Below this, NaN.
        scale: multiplicative calibration factor applied after estimation.
            The AR estimator captures *effective* spread (including short-term
            price impact) and is empirically biased upward by 1.5-3x relative
            to quoted spreads on low-volume markets. Set to e.g. 0.5 to
            calibrate down toward quoted-spread levels. Default 1.0 (no
            adjustment) is the most conservative choice.
        min_bps_per_side: floor applied after estimation (and used to
            replace NaN). Real markets always have friction; default 5 bps
            per side avoids degenerate zero-cost trades.

    Returns:
        date x ticker DataFrame of per-side spread in bps. Aligned to the
        index of `close`. Values are forward-filled within ticker so that
        non-trading days inherit the most recent estimate.
    """
    # Align inputs to the union of indexes/columns
    idx = close.index
    cols = close.columns
    high = high.reindex(index=idx, columns=cols)
    low = low.reindex(index=idx, columns=cols)
    close = close.reindex(index=idx, columns=cols)

    # Require strictly positive H, L, C; otherwise treat as missing
    valid = (high > 0) & (low > 0) & (close > 0)
    h = np.log(high.where(valid))
    l = np.log(low.where(valid))
    c = np.log(close.where(valid))
    eta = (h + l) / 2.0

    # Per-day estimator uses eta_{t+1}; shift eta forward by 1
    eta_next = eta.shift(-1)
    raw = (c - eta) * (c - eta_next)
    raw = raw.clip(lower=0.0)
    s_daily = 2.0 * np.sqrt(raw)  # effective spread as fraction of price

    # Trailing-window mean over valid observations
    s_window = s_daily.rolling(window=window, min_periods=min_obs).mean()

    # Convert to per-side bps. Effective spread is the round-trip cost as a
    # fraction of price; per side is half.
    bps = (s_window / 2.0) * 1e4 * scale

    # Forward-fill within ticker so weekends/holidays inherit the most
    # recent estimate, then floor.
    bps = bps.ffill()
    bps = bps.where(bps >= min_bps_per_side, other=min_bps_per_side)
    bps = bps.fillna(min_bps_per_side)

    return bps
