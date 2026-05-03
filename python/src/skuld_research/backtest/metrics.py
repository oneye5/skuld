"""Pure maths utilities for backtest performance metrics."""
from __future__ import annotations

import pandas as pd


def compute_max_drawdown(returns: pd.Series) -> float:
    """Max drawdown from a monthly returns series. Returns a value ≤ 0."""
    if returns.empty:
        return 0.0
    nav = (1 + returns).cumprod()
    running_max = nav.cummax()
    dd = nav / running_max - 1
    return float(dd.min())


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Rolling drawdown from peak for a monthly returns series (values ≤ 0)."""
    if returns.empty:
        return pd.Series(dtype=float)
    nav = (1 + returns).cumprod()
    running_max = nav.cummax()
    return (nav / running_max - 1).rename("drawdown")
