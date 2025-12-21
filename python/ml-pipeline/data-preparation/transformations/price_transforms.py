"""Module for converting raw prices to returns - prevents model learning wrong patterns."""

import pandas as pd
import numpy as np

from config.column_names import TIMESTAMP, TICKER, CLOSE, OPEN, HIGH, LOW, VOLUME


# Price columns that should be converted to returns
PRICE_COLUMNS = [CLOSE, OPEN, HIGH, LOW]


def convert_prices_to_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw price columns to percentage returns.
    
    Raw prices cause issues because:
    1. Higher priced stocks aren't necessarily better investments
    2. The model learns "high price = already went up = less likely to continue"
    3. This inverts the prediction signal
    
    Instead we use:
    - Daily returns (pct change from previous day)
    - Intraday range (high-low)/close
    - Gap (open-prev_close)/prev_close
    
    Args:
        df: DataFrame with OHLCV columns per ticker.
    
    Returns:
        DataFrame with price columns replaced by return columns.
    """
    if CLOSE not in df.columns:
        return df
    
    result = df.copy()
    
    # Initialize new columns
    result['return_daily'] = np.nan
    result['range_pct'] = np.nan
    result['gap_pct'] = np.nan
    result['volume_change'] = np.nan
    
    # Process each ticker separately to calculate proper returns
    for ticker in df[TICKER].unique():
        mask = df[TICKER] == ticker
        ticker_data = df.loc[mask].sort_values(TIMESTAMP)
        idx = ticker_data.index
        
        if len(ticker_data) < 2:
            continue
        
        close = ticker_data[CLOSE].values
        
        # Daily return: (close_t - close_t-1) / close_t-1 * 100
        daily_return = np.zeros(len(close))
        daily_return[1:] = (close[1:] - close[:-1]) / np.maximum(np.abs(close[:-1]), 0.0001) * 100
        daily_return[0] = 0  # First day has no return
        result.loc[idx, 'return_daily'] = daily_return
        
        # Intraday range: (high - low) / close * 100
        if HIGH in df.columns and LOW in df.columns:
            high = ticker_data[HIGH].values
            low = ticker_data[LOW].values
            range_pct = (high - low) / np.maximum(np.abs(close), 0.0001) * 100
            result.loc[idx, 'range_pct'] = range_pct
        
        # Gap: (open - prev_close) / prev_close * 100
        if OPEN in df.columns:
            open_price = ticker_data[OPEN].values
            gap = np.zeros(len(close))
            gap[1:] = (open_price[1:] - close[:-1]) / np.maximum(np.abs(close[:-1]), 0.0001) * 100
            result.loc[idx, 'gap_pct'] = gap
        
        # Volume change (log ratio to previous day)
        if VOLUME in df.columns:
            volume = ticker_data[VOLUME].values.astype(float)
            vol_change = np.zeros(len(volume))
            # Use log ratio for volume changes
            prev_vol = np.maximum(volume[:-1], 1)
            curr_vol = np.maximum(volume[1:], 1)
            vol_change[1:] = np.log(curr_vol / prev_vol)
            result.loc[idx, 'volume_change'] = vol_change
    
    # Drop raw price columns (they cause anti-prediction)
    cols_to_drop = [c for c in PRICE_COLUMNS if c in result.columns]
    result = result.drop(columns=cols_to_drop)
    
    # Also drop raw volume (keep volume_change instead)
    if VOLUME in result.columns:
        result = result.drop(columns=[VOLUME])
    
    return result


def normalize_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert macro features to change rates instead of absolute values.
    
    Absolute macro values (like GDP in millions) create similar problems
    to raw prices - they trend over time and the model learns wrong patterns.
    
    Args:
        df: DataFrame with macro columns (starting with MACRO_).
    
    Returns:
        DataFrame with macro columns converted to pct change.
    """
    result = df.copy()
    
    macro_cols = [c for c in df.columns if c.startswith('MACRO_')]
    
    if not macro_cols:
        return result
    
    # For macro data, calculate year-over-year change
    # Group by timestamp to get proper ordering
    sorted_df = result.sort_values(TIMESTAMP)
    
    for col in macro_cols:
        values = sorted_df[col].values
        
        # Calculate percentage change from previous non-null value
        pct_change = np.zeros(len(values))
        last_valid = None
        last_valid_idx = 0
        
        for i, val in enumerate(values):
            if pd.notna(val):
                if last_valid is not None and last_valid != 0:
                    pct_change[i] = (val - last_valid) / abs(last_valid) * 100
                last_valid = val
                last_valid_idx = i
        
        # Replace absolute values with changes
        result[col] = pct_change[result.index.get_indexer(sorted_df.index)]
    
    return result
