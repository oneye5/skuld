"""Cross-sectional features (ranking across tickers).

IMPORTANT: These features compute percentile ranks WITHIN each timestamp,
meaning they compare stocks to their peers at the same point in time.
This is safe from lookahead bias as long as this function is called
AFTER train/test split (which the ranking_pipeline does correctly).

The ranking is done per-timestamp, so:
- Train data ranks are computed only using train data
- Test data ranks are computed only using test data
"""

import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional rank features.
    
    Calculates the percentile rank of key features across all tickers
    for each timestamp. This helps normalize for market conditions
    (e.g., high RSI in a crash vs bull market).
    
    LEAKAGE NOTE: This function is safe from lookahead bias because:
    1. Ranking is done per-timestamp (groupby TIMESTAMP)
    2. It should be called AFTER train/test split in the pipeline
    
    Args:
        df: Wide format DataFrame with technical features.
            Should be either train or test data, NOT the full dataset.
    
    Returns:
        DataFrame with Rank_* features added (values 0.0 to 1.0).
    """
    # Features to rank - includes base technical and alpha factors
    features_to_rank = [
        # Base technical
        "RSI_14",
        "ROC_252",
        "Vol_252",
        "Dist_MA_200",
        "Pos_52w_Range",
        "NATR_14",
        "BB_Width_20",
        # Alpha factors - reversal
        "Rev_5d",
        "Rev_10d",
        # Alpha factors - momentum quality
        "Trend_RSq_60",
        "QualMom_60",
        # Alpha factors - idiosyncratic volatility
        "IdioVol_20",
        "IdioVol_60",
        # Alpha factors - information discreteness
        "InfoDisc_21",
        "InfoDisc_63",
        # Alpha factors - max effect
        "MAX_21d",
        "MaxMinSpread_21d",
        # Alpha factors - higher moments
        "Skew_60d",
        "Kurt_60d",
        "DownVol_60d",
        # Alpha factors - volume
        "RelVol_20d",
        "Amihud_21d",
        # Alpha factors - momentum acceleration
        "MomAccel_21_63",
        "Near52wHigh",
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
