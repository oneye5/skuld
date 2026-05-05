"""Operating-cash-flow-to-assets factor.

Computes operating cash flow divided by total assets using the most recent
available annual observations strictly before the rebalance date. Higher
values indicate stronger cash generation relative to the asset base.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PreparedPanel


class OcfToAssetsFactor:
    """Cross-sectional operating-cash-flow-to-assets quality signal."""

    name: str = "ocf_to_assets"

    def score(
        self,
        panel: PreparedPanel,
        t: pd.Timestamp,
        universe: list[str],
    ) -> pd.Series:
        fundamentals = panel.fundamentals
        t_naive = t.tz_localize(None) if t.tzinfo else t

        required = {
            "annual_cash_flowsfromusedin_operating_activities_direct",
            "annual_total_assets",
        }
        if fundamentals.empty or not required.issubset(fundamentals.columns):
            return pd.Series(np.nan, index=universe, dtype=float, name=self.name)

        scores: dict[str, float] = {}
        ticker_levels = fundamentals.index.get_level_values("ticker")
        for ticker in universe:
            if ticker not in ticker_levels:
                scores[ticker] = np.nan
                continue

            ticker_rows = fundamentals.xs(ticker, level="ticker")
            ticker_rows = ticker_rows[ticker_rows.index < t_naive]
            if ticker_rows.empty:
                scores[ticker] = np.nan
                continue

            ocf_series = ticker_rows[
                "annual_cash_flowsfromusedin_operating_activities_direct"
            ].dropna()
            assets_series = ticker_rows["annual_total_assets"].dropna()
            if ocf_series.empty or assets_series.empty:
                scores[ticker] = np.nan
                continue

            ocf = ocf_series.iloc[-1]
            assets = assets_series.iloc[-1]
            if pd.isna(ocf) or pd.isna(assets) or assets <= 0:
                scores[ticker] = np.nan
                continue

            scores[ticker] = float(ocf / assets)

        return pd.Series(scores, index=universe, dtype=float, name=self.name).reindex(universe)
