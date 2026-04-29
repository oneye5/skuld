"""12-1 momentum factor.

Computes the 12-month cumulative return excluding the most recent month
(the "skip month"), which is the standard academic and practitioner
momentum signal.

Reference: Asness, Moskowitz & Pedersen, "Value and Momentum Everywhere",
Journal of Finance 2013.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PreparedPanel


class MomentumFactor:
    """Cross-sectional 12-1 momentum signal.

    At rebalance date ``t``, uses the 12 monthly return periods ending one
    month before ``t`` (the skip month is the most recent available month-end
    before ``t``).  Tickers with fewer than ``min_months`` non-NaN monthly
    returns in the 12-month window are excluded (score = NaN).

    The PIT guarantee is belt-and-suspenders: the ``PreparedPanel`` is already
    built from a ``PITSnapshot`` that enforces no future data, and this class
    additionally restricts ``returns_monthly`` to dates strictly before ``t``.
    """

    name: str = "momentum"

    def __init__(self, min_months: int = 11, smoothing_months: int = 1) -> None:
        """
        Args:
            min_months: Minimum number of valid monthly return observations
                required in the 12-month window. Tickers below this threshold
                are excluded from ranking (NaN). Default 11 (per-spec).
            smoothing_months: Number of trailing rebalance dates (including ``t``)
                over which to average the raw 12-1 score. ``1`` (default) preserves
                the historical single-period behavior. ``N>1`` averages the raw
                score at ``t`` with the raw scores at the ``N-1`` most recent
                rebalance month-ends strictly before ``t``. Per-ticker mean is
                taken over non-NaN raw scores only; a ticker with zero non-NaN
                raw scores in the window is NaN.
        """
        if smoothing_months < 1:
            raise ValueError(f"smoothing_months must be >= 1, got {smoothing_months}")
        self.min_months = min_months
        self.smoothing_months = smoothing_months

    def score(
        self,
        panel: PreparedPanel,
        t: pd.Timestamp,
        universe: list[str],
    ) -> pd.Series:
        """Compute (optionally smoothed) 12-1 momentum scores at rebalance date ``t``.

        With ``smoothing_months == 1`` returns the raw 12-1 score at ``t``. With
        ``smoothing_months == N > 1``, averages the raw scores at ``t`` and the
        ``N-1`` most recent rebalance month-ends strictly before ``t`` (taken
        from ``panel.universe_mask.index``). When fewer than ``N`` such dates
        exist, averages over what is available.

        Args:
            panel: PreparedPanel with ``returns_monthly`` available.
            t: Rebalance date. Only monthly periods strictly before each
                averaged ``t_i`` are used (PIT-safe).
            universe: Tickers to score.

        Returns:
            Series[float64] indexed by ``universe``. NaN where insufficient history.
        """
        if self.smoothing_months == 1:
            return self._score_raw(panel, t, universe)

        t_naive = t.tz_localize(None) if t.tzinfo else t
        rebalance_dates = panel.universe_mask.index
        prior = rebalance_dates[rebalance_dates < t_naive]
        # Take the most recent (N-1) prior rebalance dates
        prior_window = prior[-(self.smoothing_months - 1):] if self.smoothing_months > 1 else prior[:0]

        score_dates = list(prior_window) + [t]
        raw_scores = [self._score_raw(panel, ts, universe) for ts in score_dates]

        stacked = pd.concat(raw_scores, axis=1)
        # Mean over non-NaN entries; tickers with all-NaN row remain NaN.
        smoothed = stacked.mean(axis=1, skipna=True)
        smoothed.name = self.name
        return smoothed.reindex(universe)

    def _score_raw(
        self,
        panel: PreparedPanel,
        t: pd.Timestamp,
        universe: list[str],
    ) -> pd.Series:
        """Raw single-period 12-1 momentum score at ``t``."""
        monthly = panel.returns_monthly
        t_naive = t.tz_localize(None) if t.tzinfo else t

        # All available month-end return periods strictly before t
        avail_idx = monthly.index[monthly.index < t_naive]

        # Need at least 2 available months: 1 skip month + at least 1 signal month
        if len(avail_idx) < 2:
            return pd.Series(np.nan, index=universe, dtype=float, name=self.name)

        # Skip the most recent available month (t-1 — the reversal month).
        # Use the up-to-12 months before that as the signal window.
        signal_idx = avail_idx[:-1][-12:]  # at most 12 months, skip most recent

        # Restrict to tickers present in the panel
        available_tickers = [tk for tk in universe if tk in monthly.columns]
        if not available_tickers:
            return pd.Series(np.nan, index=universe, dtype=float, name=self.name)

        window = monthly.loc[signal_idx, available_tickers]

        # Count valid (non-NaN) monthly return observations per ticker
        valid_counts = window.notna().sum()

        # Cumulative return: fill NaN months with 0 (neutral return) so that
        # the cumulative product spans the same calendar window for all tickers,
        # making scores directly comparable across tickers with minor data gaps.
        # Tickers with too few valid months are masked to NaN afterwards.
        cum_returns = (1.0 + window.fillna(0.0)).prod() - 1.0

        # Apply minimum-history mask
        scores = cum_returns.where(valid_counts >= self.min_months)

        return scores.reindex(universe)
