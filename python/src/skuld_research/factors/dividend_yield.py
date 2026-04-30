"""Trailing dividend-yield factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PreparedPanel


class DividendYieldFactor:
    """Rank tickers by trailing dividends divided by last adjusted close."""

    name: str = "dividend_yield"

    def __init__(self, lookback_months: int = 12, min_dividends: int = 1) -> None:
        if lookback_months < 1:
            raise ValueError(f"lookback_months must be >= 1, got {lookback_months}")
        if min_dividends < 1:
            raise ValueError(f"min_dividends must be >= 1, got {min_dividends}")
        self.lookback_months = lookback_months
        self.min_dividends = min_dividends

    def score(
        self,
        panel: PreparedPanel,
        t: pd.Timestamp,
        universe: list[str],
    ) -> pd.Series:
        """Compute trailing dividend yield using only observations before ``t``."""
        t_naive = t.tz_localize(None) if t.tzinfo else t
        prices = panel.prices
        actions = panel.corporate_actions
        if prices.empty or actions.empty:
            return pd.Series(np.nan, index=universe, dtype=float, name=self.name)

        prior_prices = prices.loc[prices.index < t_naive]
        if prior_prices.empty:
            return pd.Series(np.nan, index=universe, dtype=float, name=self.name)

        last_price = prior_prices.reindex(columns=universe).ffill().iloc[-1]
        start = t_naive - pd.DateOffset(months=self.lookback_months)
        divs = actions.loc[
            (actions["type"] == "dividend")
            & (actions["ex_date"] >= start)
            & (actions["ex_date"] < t_naive)
            & (actions["ticker"].isin(universe))
        ]
        if divs.empty:
            return pd.Series(np.nan, index=universe, dtype=float, name=self.name)

        sums = divs.groupby("ticker")["factor"].sum()
        counts = divs.groupby("ticker")["factor"].size()
        scores = sums / last_price.reindex(sums.index)
        scores = scores.where(counts >= self.min_dividends)
        scores = scores.where(last_price.reindex(scores.index) > 0.0)
        scores.name = self.name
        return scores.reindex(universe)
