"""Module for generating target labels for price prediction."""

import pandas as pd
import numpy as np

from config.column_names import TIMESTAMP, TICKER, CLOSE, TARGET
from config.model_config import LOOKAHEAD_DAYS, GAIN_THRESHOLD_PCT, MS_PER_DAY


def create_labels(
    df: pd.DataFrame,
    lookahead_days: int = LOOKAHEAD_DAYS,
    gain_threshold_pct: float = GAIN_THRESHOLD_PCT,
    price_lookup_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Create binary target labels based on future price change.
    
    For each observation, calculates if the price gains >= threshold% within
    lookahead_days. Rows where future price cannot be determined are dropped.
    
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
    
    # Process each ticker separately using vectorized operations
    result_dfs = []
    
    for ticker in df[TICKER].unique():
        ticker_df = df[df[TICKER] == ticker].copy()  # Only copy filtered data
        ticker_df.sort_values(TIMESTAMP, inplace=True)
        ticker_df.reset_index(drop=True, inplace=True)
        
        if CLOSE not in ticker_df.columns:
            continue
        
        # Get price lookup data for this ticker (may include future data)
        ticker_lookup = lookup_df[lookup_df[TICKER] == ticker].sort_values(TIMESTAMP)
        
        if CLOSE not in ticker_lookup.columns:
            continue
        
        # Calculate target timestamp for each row
        target_ts = ticker_df[TIMESTAMP] + lookahead_ms
        
        # Vectorized lookup using merge_asof (much faster than row-by-row)
        # Create lookup DataFrame with target timestamps
        lookup_df_for_merge = ticker_lookup[[TIMESTAMP, CLOSE]].copy()
        lookup_df_for_merge = lookup_df_for_merge.sort_values(TIMESTAMP)
        
        target_df = pd.DataFrame({
            'target_ts': target_ts.values,
            'orig_idx': ticker_df.index.values
        }).sort_values('target_ts')
        
        # merge_asof finds the first timestamp >= target_ts (direction='forward')
        merged = pd.merge_asof(
            target_df,
            lookup_df_for_merge,
            left_on='target_ts',
            right_on=TIMESTAMP,
            direction='forward'
        )
        
        # Reorder to match original index
        merged = merged.set_index('orig_idx').reindex(ticker_df.index)
        ticker_df['future_close'] = merged[CLOSE].values
        
        # Calculate percentage change
        current_price = ticker_df[CLOSE]
        future_price = ticker_df['future_close']
        
        pct_change = ((future_price - current_price) / current_price) * 100
        
        # Assign target
        ticker_df[TARGET] = (pct_change >= gain_threshold_pct).astype(float)
        ticker_df.loc[ticker_df['future_close'].isna(), TARGET] = np.nan
        
        # Drop the temporary column
        ticker_df = ticker_df.drop(columns=['future_close'])
        
        result_dfs.append(ticker_df)
    
    if not result_dfs:
        df[TARGET] = np.nan
        return df.dropna(subset=[TARGET])
    
    result = pd.concat(result_dfs, ignore_index=True)
    
    # Drop rows without valid target
    result = result.dropna(subset=[TARGET])
    result[TARGET] = result[TARGET].astype(int)
    
    return result
