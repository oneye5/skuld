"""Low-volatility factor.

Computes the trailing 12-month realised daily-return volatility (standard
deviation) and scores as -vol, so lower-volatility tickers rank higher.

This is a standard defensive factor in the literature; low-volatility stocks
tend to outperform high-volatility stocks on a risk-adjusted basis (the
"low-volatility anomaly").

Reference: Baker, Bradley & Wurgler, "Benchmarks as Limits to Arbitrage:
Understanding the Low-Volatility Anomaly", Financial Analysts Journal 2011.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PreparedPanel


class LowVolatilityFactor:
    """Cross-sectional low-volatility signal.

    At rebalance date ``t``, computes the trailing realised volatility of
    daily returns over the past ``lookback_months`` months (default 12).
    Tickers with fewer than ``min_months`` of non-NaN daily returns in the
    window are excluded (score = NaN).

    Score = ``-vol`` so that lower-volatility tickers rank higher (higher score
    = more attractive). This composes naturally in z-space with other factors
    in the combiner.

    The PIT guarantee is belt-and-suspenders: the ``PreparedPanel`` is already
    built from a ``PITSnapshot`` that enforces no future data, and this class
    additionally restricts ``returns_daily`` to dates strictly before ``t``.
    """

    name: str = "low_volatility"

    def __init__(self, lookback_months: int = 12, min_months: int = 6) -> None:
        """
        Args:
            lookback_months: Number of months of daily returns to use for vol
                calculation. Default 12 (one year).
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
        """Compute low-volatility scores for ``universe`` at rebalance date ``t``.

        Args:
            panel: PreparedPanel with ``returns_daily`` available.
            t: Rebalance date. Only daily periods strictly before ``t`` are used.
            universe: Tickers to score.

        Returns:
            Series[float64] indexed by ``universe``. NaN where insufficient history.
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

        # Compute standard deviation of daily returns (volatility)
        # Use ddof=1 for sample standard deviation
        vol = window.std(ddof=1)

        # Score is -vol (lower volatility = higher score)
        scores = -vol

        # Apply minimum-history mask: tickers with < min_days valid observations get NaN
        scores = scores.where(valid_counts >= min_days)

        return scores.reindex(universe)
