"""Book-to-market factor.

Computes common-equity book value divided by market capitalisation using the
most recent available annual stockholders' equity strictly before the rebalance
date. Higher book-to-market indicates a cheaper / more value-like stock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PreparedPanel


class BookToMarketFactor:
    """Cross-sectional book-to-market value signal."""

    name: str = "book_to_market"

    def score(
        self,
        panel: PreparedPanel,
        t: pd.Timestamp,
        universe: list[str],
    ) -> pd.Series:
        fundamentals = panel.fundamentals
        market_cap = panel.market_cap
        t_naive = t.tz_localize(None) if t.tzinfo else t

        if fundamentals.empty or "annual_stockholders_equity" not in fundamentals.columns:
            return pd.Series(np.nan, index=universe, dtype=float, name=self.name)

        avail_idx = market_cap.index[market_cap.index < t_naive]
        if len(avail_idx) == 0:
            return pd.Series(np.nan, index=universe, dtype=float, name=self.name)

        available_tickers = [tk for tk in universe if tk in market_cap.columns]
        if not available_tickers:
            return pd.Series(np.nan, index=universe, dtype=float, name=self.name)

        latest_market_cap = market_cap.loc[avail_idx, available_tickers].iloc[-1]
        equity_series = fundamentals["annual_stockholders_equity"].dropna()

        scores: dict[str, float] = {}
        ticker_level = equity_series.index.get_level_values("ticker")
        for ticker in available_tickers:
            if ticker not in ticker_level:
                scores[ticker] = np.nan
                continue

            ticker_equity = equity_series.xs(ticker, level="ticker")
            ticker_equity = ticker_equity[ticker_equity.index < t_naive]
            if ticker_equity.empty:
                scores[ticker] = np.nan
                continue

            latest_book = ticker_equity.iloc[-1]
            latest_cap = latest_market_cap.get(ticker)
            if pd.isna(latest_cap) or latest_cap <= 0:
                scores[ticker] = np.nan
                continue

            scores[ticker] = float(latest_book / latest_cap)

        return pd.Series(scores, index=universe, dtype=float, name=self.name).reindex(universe)
