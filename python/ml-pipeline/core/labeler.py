"""Module for generating target labels for price prediction."""

import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER, CLOSE, TARGET
from config.settings import LOOKAHEAD_DAYS, GAIN_THRESHOLD_PCT, MS_PER_DAY


def create_labels(
    df: pd.DataFrame,
    lookahead_days: int = LOOKAHEAD_DAYS,
    gain_threshold_pct: float = GAIN_THRESHOLD_PCT,
    price_lookup_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create binary target labels based on future price change.
    
    For each observation, calculates if the price gains >= threshold% within
    lookahead_days. Rows where future price cannot be determined are dropped.
    
    The automatic cutoff ensures that only rows with valid future data for
    labeling are included. Rows near the end of the dataset that don't have
    enough future data are automatically excluded.
    
    Args:
        df: Wide format DataFrame with timestamp, ticker, and Close columns.
        lookahead_days: Number of days to look ahead for price change.
        gain_threshold_pct: Minimum percentage gain for positive class.
        price_lookup_df: Optional DataFrame with future price data for lookup.
                        If None, uses df for both rows to label and price lookup.
                        Use this when labeling test data that needs future prices
                        beyond the test period.
    
    Returns:
        DataFrame with target column added. Rows without valid target are dropped.
    """
    lookahead_ms = lookahead_days * MS_PER_DAY
    
    # Use provided price_lookup_df or fall back to df
    lookup_df = price_lookup_df if price_lookup_df is not None else df
    
    result_dfs = []
    
    for ticker in df[TICKER].unique():
        ticker_df = df[df[TICKER] == ticker].copy()
        ticker_df = ticker_df.sort_values(TIMESTAMP).reset_index(drop=True)
        
        if CLOSE not in ticker_df.columns:
            continue
        
        # Get price lookup data for this ticker
        ticker_lookup = lookup_df[lookup_df[TICKER] == ticker].sort_values(TIMESTAMP)
        
        if CLOSE not in ticker_lookup.columns:
            continue
        
        # Calculate target timestamp for each row
        target_ts = ticker_df[TIMESTAMP] + lookahead_ms
        
        # Create lookup DataFrame
        lookup_for_merge = ticker_lookup[[TIMESTAMP, CLOSE]].copy()
        lookup_for_merge = lookup_for_merge.sort_values(TIMESTAMP)
        
        target_df = pd.DataFrame({
            "target_ts": target_ts.values,
            "orig_idx": ticker_df.index.values,
        }).sort_values("target_ts")
        
        # merge_asof finds the first timestamp >= target_ts (direction='forward')
        merged = pd.merge_asof(
            target_df,
            lookup_for_merge,
            left_on="target_ts",
            right_on=TIMESTAMP,
            direction="forward",
        )
        
        # Reorder to match original index
        merged = merged.set_index("orig_idx").reindex(ticker_df.index)
        ticker_df["future_close"] = merged[CLOSE].values
        
        # Calculate percentage change
        current_price = ticker_df[CLOSE]
        future_price = ticker_df["future_close"]
        
        pct_change = ((future_price - current_price) / current_price) * 100
        
        # Assign target
        ticker_df[TARGET] = (pct_change >= gain_threshold_pct).astype(float)
        ticker_df.loc[ticker_df["future_close"].isna(), TARGET] = np.nan
        
        # Drop the temporary column
        ticker_df = ticker_df.drop(columns=["future_close"])
        
        result_dfs.append(ticker_df)
    
    if not result_dfs:
        df = df.copy()
        df[TARGET] = np.nan
        return df.dropna(subset=[TARGET])
    
    result = pd.concat(result_dfs, ignore_index=True)
    
    # Drop rows without valid target (automatic cutoff)
    result = result.dropna(subset=[TARGET])
    result[TARGET] = result[TARGET].astype(int)
    
    return result


def get_max_labelable_timestamp(
    max_data_timestamp: int,
    lookahead_days: int = LOOKAHEAD_DAYS,
) -> int:
    """Calculate the maximum timestamp that can be labeled.
    
    Data points after this timestamp won't have enough future data
    for labeling and should be excluded from train/test.
    
    Args:
        max_data_timestamp: Maximum timestamp in the dataset.
        lookahead_days: Days needed for lookahead.
    
    Returns:
        Maximum timestamp that can receive a valid label.
    """
    lookahead_ms = lookahead_days * MS_PER_DAY
    return max_data_timestamp - lookahead_ms
