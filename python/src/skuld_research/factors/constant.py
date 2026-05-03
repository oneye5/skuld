"""Constant signal generator (used only by benchmarks)."""
from __future__ import annotations

import pandas as pd

from skuld_common.contracts import PreparedPanel
from skuld_research.factors.protocols import SignalGenerator


class ConstantOneSignal(SignalGenerator):
    """Returns +1.0 for every name in the universe at every rebalance date.
    
    Used only by the equal-weighted benchmark to produce uniform cross-sectional
    scores that the optimizer then weights equally.
    """

    @property
    def name(self) -> str:
        return "constant_one"

    def score(
        self,
        panel: PreparedPanel,
        asof: pd.Timestamp,
        universe: list[str],
    ) -> pd.Series:
        """Return +1.0 for every ticker in universe.
        
        Args:
            panel: PreparedPanel (unused).
            asof: rebalance date (unused).
            universe: list of ticker symbols to score.
        
        Returns:
            Series of +1.0 values indexed by universe tickers.
        """
        return pd.Series(1.0, index=universe)
