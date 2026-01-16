"""Attention-based aggregate features from Wikipedia pageviews.

This module computes aggregate attention features (fear indicators, etc.)
that combine multiple Wikipedia pageview sources.

For ticker-specific lag/MA/momentum features on Wiki_Views, see:
    - features/lag_ma_config.py - Configuration for which features get lags/MAs
    - features/lag_ma_features.py - Generic lag/MA computation

Research basis:
- Moat et al. (2013) - "Quantifying Wikipedia Usage Patterns Before Stock Market Moves"
  Found that Wikipedia company page views predict stock price movements.
- Preis et al. (2013) - "Quantifying Trading Behavior in Financial Markets Using Google Trends"
  Showed searches for crisis-related terms predict market declines.

Aggregate features computed:
- Fear indicators (sum/mean of crisis-related page views)
- Fear momentum and spikes
- Mobile vs desktop ratio (for ticker-specific attention)
"""

import pandas as pd
import numpy as np
from typing import List, Optional

from config.columns import TICKER, TIMESTAMP
from config.settings import EPSILON


# Column patterns for Wiki features
WIKI_VIEWS_PATTERN = "Wiki_Views"
WIKI_DESKTOP_PATTERN = "Wiki_Views_Desktop"
WIKI_MOBILE_PATTERN = "Wiki_Views_Mobile"


def add_mobile_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add mobile vs desktop attention ratio features.
    
    This is a special feature not covered by the generic lag/MA system
    because it's a ratio of two columns.
    
    Args:
        df: Wide format DataFrame with Wiki_Views_Desktop and Wiki_Views_Mobile.
    
    Returns:
        DataFrame with mobile ratio features added.
    """
    if WIKI_DESKTOP_PATTERN not in df.columns or WIKI_MOBILE_PATTERN not in df.columns:
        return df
    
    if TICKER not in df.columns:
        return df
    
    df = df.copy()
    
    # Mobile vs Desktop ratio
    df["Attn_Mobile_Ratio"] = df[WIKI_MOBILE_PATTERN] / (
        df[WIKI_DESKTOP_PATTERN] + df[WIKI_MOBILE_PATTERN] + EPSILON
    )
    
    # Change in mobile ratio over 14 days
    df["Attn_Mobile_Ratio_Change_14"] = df.groupby(TICKER)["Attn_Mobile_Ratio"].transform(
        lambda x: x.diff(14)
    )
    
    return df


def add_aggregate_attention_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add aggregate attention features across all macro Wiki columns.
    
    These capture overall market attention/fear levels:
    - Total views across fear-related pages
    - Average spike across sentiment indicators
    - Fear momentum and spike indicators
    
    Also adds mobile ratio features for ticker-specific attention.
    
    Args:
        df: DataFrame with macro Wiki columns.
    
    Returns:
        DataFrame with aggregate attention features.
    """
    # Add mobile ratio features first
    df = add_mobile_ratio_features(df)
    
    # Find fear/crisis related columns
    fear_patterns = [
        "Recession", "Financial_crisis", "Stock_market_crash",
        "Credit_crunch", "Economic_bubble", "Bank_run",
        "Inflation", "Unemployment", "Debt_crisis",
    ]
    
    fear_cols = []
    for col in df.columns:
        if "Wiki_Views" in col and any(p.lower() in col.lower() for p in fear_patterns):
            fear_cols.append(col)
    
    if not fear_cols:
        return df
    
    df = df.copy()
    
    # Aggregate fear attention
    df["Attn_Fear_Total"] = df[fear_cols].sum(axis=1)
    df["Attn_Fear_Mean"] = df[fear_cols].mean(axis=1)
    
    # Fear momentum (7-day change in total fear attention)
    df = df.sort_values(TIMESTAMP)
    df["Attn_Fear_Mom_7"] = df["Attn_Fear_Total"].pct_change(periods=7, fill_method=None)
    df["Attn_Fear_Mom_14"] = df["Attn_Fear_Total"].pct_change(periods=14, fill_method=None)
    
    # Fear spike (current vs 28-day average)
    fear_ma_28 = df["Attn_Fear_Total"].rolling(window=28, min_periods=7).mean()
    df["Attn_Fear_Spike"] = df["Attn_Fear_Total"] / (fear_ma_28 + EPSILON)
    
    # Restore sort order
    if TICKER in df.columns:
        df = df.sort_values([TICKER, TIMESTAMP])
    
    return df