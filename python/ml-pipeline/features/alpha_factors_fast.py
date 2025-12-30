"""Fast vectorized alpha factors using groupby operations.

This module provides a FAST version of alpha factors that uses pandas groupby
instead of per-ticker Python loops. It skips the most expensive calculations
(like rolling R-squared) in favor of speed.

Performance: ~5s vs ~120s for original alpha_factors.py
"""

import pandas as pd
import numpy as np

from config.columns import CLOSE, HIGH, LOW, VOLUME, TICKER, TIMESTAMP
from config.settings import EPSILON


def add_alpha_factors_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Add fast vectorized alpha factors using groupby operations.
    
    This is optimized for speed over comprehensiveness.
    Only adds the most impactful features.
    
    Args:
        df: Wide format DataFrame with OHLCV columns.
    
    Returns:
        DataFrame with alpha factor features added.
    """
    if TICKER not in df.columns or CLOSE not in df.columns:
        return df
    
    df = df.sort_values([TICKER, TIMESTAMP]).copy()
    
    # Pre-compute returns once
    df["_ret"] = df.groupby(TICKER, sort=False)[CLOSE].pct_change()
    
    # --- Short-term Reversal ---
    df["Rev_5d"] = df.groupby(TICKER, sort=False)[CLOSE].pct_change(5)
    df["Rev_10d"] = df.groupby(TICKER, sort=False)[CLOSE].pct_change(10)
    df["Rev_21d"] = df.groupby(TICKER, sort=False)[CLOSE].pct_change(21)
    
    # --- Volatility (use ewm for speed) ---
    df["Vol_20d"] = df.groupby(TICKER, sort=False)["_ret"].transform(
        lambda x: x.ewm(span=20, min_periods=10).std()
    )
    
    # --- Max/Min returns ---
    df["MAX_21d"] = df.groupby(TICKER, sort=False)["_ret"].transform(
        lambda x: x.rolling(21, min_periods=10).max()
    )
    df["MIN_21d"] = df.groupby(TICKER, sort=False)["_ret"].transform(
        lambda x: x.rolling(21, min_periods=10).min()
    )
    df["MaxMinSpread_21d"] = df["MAX_21d"] - df["MIN_21d"]
    
    # --- Momentum acceleration ---
    mom_21 = df.groupby(TICKER, sort=False)[CLOSE].pct_change(21)
    mom_63 = df.groupby(TICKER, sort=False)[CLOSE].pct_change(63)
    df["MomAccel_21_63"] = mom_21 - mom_63 / 3
    
    # --- Near 52-week high (use ewm max approximation for speed) ---
    df["Near52wHigh"] = df[CLOSE] / df.groupby(TICKER, sort=False)[CLOSE].transform(
        lambda x: x.rolling(252, min_periods=60).max()
    ).replace(0, np.nan).fillna(1)
    
    # Cleanup
    df = df.drop(columns=["_ret"], errors="ignore")
    
    return df.sort_values(TIMESTAMP).reset_index(drop=True)
