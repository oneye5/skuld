"""Factor signal generator protocol for Skuld research pipeline."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from skuld_common.contracts import PreparedPanel


@runtime_checkable
class SignalGenerator(Protocol):
    """Contract for per-factor signal generators.

    Each factor produces a cross-sectional score for tickers in the universe
    at a given point-in-time. Scores are raw (not normalised) — the combiner
    handles z-scoring, winsorisation, and shrinkage.

    Implementors must set the ``name`` class/instance attribute to a stable
    snake_case string that identifies the factor (e.g. ``"momentum"``).
    """

    name: str

    def score(
        self,
        panel: PreparedPanel,
        t: pd.Timestamp,
        universe: list[str],
    ) -> pd.Series:
        """Compute cross-sectional factor scores.

        Args:
            panel: PreparedPanel (already PIT-safe for ``panel.asof``).
            t: Rebalance date. Only data strictly before ``t`` may be used.
            universe: Tickers eligible for ranking on this date.

        Returns:
            Series[float64] indexed by ``universe``. NaN for tickers that
            cannot be scored (insufficient history, missing data, etc.).
        """
        ...
