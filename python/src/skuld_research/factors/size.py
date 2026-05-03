"""Size factor.

Computes the negative log-market-cap score with winsorisation to cap the
influence of extreme micro-caps and mega-caps. Smaller-cap tickers rank higher
(the "size effect" or "small-cap premium").

The factor score is ``-log(market_cap)`` after winsorisation at p1 and p99 of
the universe-wide distribution. Winsorisation is applied to the raw score
(post-log) to prevent extreme values from dominating.

Reference: Fama & French, "Common risk factors in the returns on stocks and
bonds", Journal of Financial Economics 1993.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PreparedPanel


class SizeFactor:
    """Cross-sectional size signal.

    At rebalance date ``t``, takes the most recent available market cap (up to
    5 days of forward-fill, matching the prepared panel's ``mc_ffill_days``
    default), computes ``-log(mcap)``, then winsorises the score at p1 and p99
    to cap extreme values.

    Score = ``-log(mcap)`` after winsorisation, so smaller-cap tickers rank
    higher (higher score = more attractive). Missing market cap results in NaN.

    The PIT guarantee is belt-and-suspenders: the ``PreparedPanel`` is already
    built from a ``PITSnapshot`` that enforces no future data, and this class
    additionally restricts ``market_cap`` to dates strictly before ``t``.
    """

    name: str = "size"

    def __init__(self) -> None:
        """No configuration parameters for the basic size factor."""
        pass

    def score(
        self,
        panel: PreparedPanel,
        t: pd.Timestamp,
        universe: list[str],
    ) -> pd.Series:
        """Compute size scores for ``universe`` at rebalance date ``t``.

        Args:
            panel: PreparedPanel with ``market_cap`` available.
            t: Rebalance date. Only market cap values strictly before ``t`` are used.
            universe: Tickers to score.

        Returns:
            Series[float64] indexed by ``universe``. NaN where market cap missing.
        """
        mcap_df = panel.market_cap
        t_naive = t.tz_localize(None) if t.tzinfo else t

        # All available market cap dates strictly before t
        avail_idx = mcap_df.index[mcap_df.index < t_naive]

        if len(avail_idx) == 0:
            # No data available before t
            return pd.Series(np.nan, index=universe, dtype=float, name=self.name)

        # Restrict to tickers present in the panel
        available_tickers = [tk for tk in universe if tk in mcap_df.columns]
        if not available_tickers:
            return pd.Series(np.nan, index=universe, dtype=float, name=self.name)

        # Take the most recent market cap before t (the prepared panel already
        # forward-filled up to mc_ffill_days, so we just take the last value)
        most_recent_mcap = mcap_df.loc[avail_idx, available_tickers].iloc[-1]

        # If majority of the universe has NaN market_cap, fall back to proxy.
        # This covers pre-2022 periods where shares-outstanding data is absent.
        if (
            most_recent_mcap.isna().mean() > 0.5
            and not panel.market_cap_proxy.empty
        ):
            proxy_df = panel.market_cap_proxy
            proxy_avail = proxy_df.index[proxy_df.index < t_naive]
            if len(proxy_avail) > 0:
                proxy_tickers = [tk for tk in available_tickers if tk in proxy_df.columns]
                if proxy_tickers:
                    proxy_row = proxy_df.loc[proxy_avail, proxy_tickers].dropna(how="all")
                    if not proxy_row.empty:
                        most_recent_mcap = proxy_row.iloc[-1].reindex(available_tickers)

        # Compute -log(mcap) where mcap is valid
        scores = -np.log(most_recent_mcap)

        # Winsorise at p1 and p99 of the universe-wide distribution
        # (only for tickers with valid scores)
        valid_scores = scores.dropna()
        if len(valid_scores) > 0:
            p1 = valid_scores.quantile(0.01)
            p99 = valid_scores.quantile(0.99)
            scores = scores.clip(lower=p1, upper=p99)

        return scores.reindex(universe)
