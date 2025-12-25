"""Cross-sectional features (ranking across tickers)."""

import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional rank features.
    
    Calculates the percentile rank of key features across all tickers
    for each timestamp. This helps normalize for market conditions
    (e.g., high RSI in a crash vs bull market).
    
    Args:
        df: Wide format DataFrame with technical features.
    
    Returns:
        DataFrame with Rank_* features added.
    """
    # Features to rank
    # We focus on the most important technical indicators
    features_to_rank = [
        "RSI_14",
        "ROC_252",
        "Vol_252",
        "Dist_MA_200",
        "Pos_52w_Range",
        "NATR_14",
        "BB_Width_20",
    ]
    
    # Only rank features that exist in the dataframe
    cols_to_rank = [c for c in features_to_rank if c in df.columns]
    
    if not cols_to_rank:
        return df
        
    result = df.copy()
    
    # Group by timestamp and rank
    # pct=True gives percentile rank (0.0 to 1.0)
    for col in cols_to_rank:
        rank_col = f"Rank_{col}"
        
        # We use transform to keep the index aligned
        result[rank_col] = result.groupby(TIMESTAMP)[col].rank(pct=True)
        
        # Fill NaN ranks (e.g. if only 1 ticker or all NaN) with 0.5
        result[rank_col] = result[rank_col].fillna(0.5)
        
    return result
