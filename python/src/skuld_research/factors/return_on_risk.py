"""Return-on-risk factor.

Computes trailing Sharpe-like signal per ticker: annualised_return / annualised_vol
over a configurable lookback window. Scores as positive Sharpe (higher = more attractive).

Rationale: Pure low-volatility selects lower absolute-return stocks in NZX, hurting
long-only portfolio returns. Return-on-risk rewards stocks with favorable return/vol
tradeoff, avoiding the absolute-return drag of the raw low-vol signal.

Lookback default: 12 months of daily returns (252 trading days).
Min obs: 6 months (126 days) of valid daily return observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PreparedPanel


class ReturnOnRiskFactor:
    """Cross-sectional return-on-risk (trailing Sharpe-like) signal.

    At rebalance date ``t``, computes the annualised return divided by
    annualised volatility of daily returns over the past ``lookback_months``
    months (default 12). Tickers with fewer than ``min_months`` of non-NaN
    daily returns in the window are excluded (score = NaN). Zero-volatility
    tickers also receive NaN to avoid infinite scores.

    Score = ``annualised_return / annualised_vol`` so that tickers with better
    risk-adjusted returns rank higher. This avoids the absolute-return drag of
    the raw low-volatility signal in a high-cost long-only environment.

    The PIT guarantee is belt-and-suspenders: the ``PreparedPanel`` is already
    built from a ``PITSnapshot`` that enforces no future data, and this class
    additionally restricts ``returns_daily`` to dates strictly before ``t``.
    """

    name: str = "return_on_risk"

    def __init__(self, lookback_months: int = 12, min_months: int = 6) -> None:
        """
        Args:
            lookback_months: Number of months of daily returns to use for
                the Sharpe-like calculation. Default 12 (one year).
            min_months: Minimum number of months (×21 trading days) of valid
                daily return observations required. Tickers below this threshold
                are excluded from ranking (NaN). Default 6.
        """
        self.lookback_months = lookback_months
        self.min_months = min_months

    def score(
        self,
        panel: PreparedPanel,
        t: pd.Timestamp,
        universe: list[str],
    ) -> pd.Series:
        """Compute return-on-risk scores for ``universe`` at rebalance date ``t``.

        Args:
            panel: PreparedPanel with ``returns_daily`` available.
            t: Rebalance date. Only daily periods strictly before ``t`` are used.
            universe: Tickers to score.

        Returns:
            Series[float64] indexed by ``universe``. NaN where insufficient history
            or zero volatility.
        """
        daily_returns = panel.returns_daily
        t_naive = t.tz_localize(None) if t.tzinfo else t

        # All available daily return periods strictly before t
        avail_idx = daily_returns.index[daily_returns.index < t_naive]

        # Calculate the lookback window in business days (assume 21 per month)
        lookback_days = self.lookback_months * 21
        min_days = self.min_months * 21

        if len(avail_idx) < min_days:
            # Not enough history at all
            return pd.Series(np.nan, index=universe, dtype=float, name=self.name)

        # Take the most recent lookback_days of available data
        window_idx = avail_idx[-lookback_days:]

        # Restrict to tickers present in the panel
        available_tickers = [tk for tk in universe if tk in daily_returns.columns]
        if not available_tickers:
            return pd.Series(np.nan, index=universe, dtype=float, name=self.name)

        window = daily_returns.loc[window_idx, available_tickers]

        # Count valid (non-NaN) daily return observations per ticker
        valid_counts = window.notna().sum()

        # Annualised return: mean daily return × 252
        mean_daily = window.mean()
        annualised_return = mean_daily * 252

        # Annualised volatility: std daily return × sqrt(252)
        # Use ddof=1 for sample standard deviation
        vol_daily = window.std(ddof=1)
        annualised_vol = vol_daily * np.sqrt(252)

        # Sharpe-like score: annualised return / annualised vol
        # Near-zero-vol tickers → NaN (not inf) via where().
        # Threshold of 1e-8 annual vol (≈5e-10 daily) catches truly constant-price
        # series while leaving all real tickers unaffected.
        scores = annualised_return / annualised_vol.where(annualised_vol > 1e-8)

        # Apply minimum-history mask: tickers with < min_days valid observations get NaN
        scores = scores.where(valid_counts >= min_days)

        return scores.reindex(universe)
