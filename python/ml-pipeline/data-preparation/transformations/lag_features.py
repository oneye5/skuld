"""Module for creating lag features from macro indicators."""

import pandas as pd
import numpy as np

from config.column_names import TIMESTAMP, TICKER, MACRO_PREFIX


def add_macro_lag_features(df: pd.DataFrame, lags: list[int] = None) -> pd.DataFrame:
    """
    Add lag features for macro indicators.
    
    Macro data often has predictive power with time delays (leading indicators).
    This creates lagged versions of macro features to capture this.
    
    Args:
        df: DataFrame with macro features (columns starting with MACRO_).
        lags: List of lag periods in days. Default: [30, 90]
    
    Returns:
        DataFrame with additional lag features for macro columns.
    """
    if lags is None:
        lags = [30, 90]  # Reduced from [30, 60, 90, 180] for memory efficiency
    
    if df.empty:
        return df
    
    # Get macro columns
    macro_cols = [c for c in df.columns if c.startswith(MACRO_PREFIX)]
    
    if not macro_cols:
        return df
    
    # Sort by timestamp
    df = df.sort_values(TIMESTAMP).reset_index(drop=True)
    
    # Calculate change features for each macro column in-place
    for col in macro_cols:
        # Rate of change for different periods
        for lag in lags:
            # Percentage change from lag periods ago
            col_name = f"{col}_change_{lag}d"
            shifted = df[col].shift(lag)
            # Safe division avoiding divide by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                change = ((df[col] - shifted) / shifted.abs().replace(0, np.nan)) * 100
            df[col_name] = change.astype('float32')
    
    return df


def add_ticker_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features for key ticker metrics.
    
    Adds lagged returns and price levels to capture momentum persistence.
    
    Args:
        df: DataFrame with ticker data.
    
    Returns:
        DataFrame with additional lag features.
    """
    if df.empty:
        return df
    
    # Define which columns to create lags for (technical features)
    # Reduced set for memory efficiency
    lag_cols = ['return_5d', 'rsi_14', 'price_to_sma_20']
    
    # Filter to columns that exist
    lag_cols = [c for c in lag_cols if c in df.columns]
    
    if not lag_cols:
        return df
    
    # Sort by ticker and timestamp
    df = df.sort_values([TICKER, TIMESTAMP]).reset_index(drop=True)
    
    # Create lag columns in-place using groupby
    for col in lag_cols:
        # Add 20-day lagged values only (reduced from 5d and 20d)
        col_name = f"{col}_lag_20d"
        df[col_name] = df.groupby(TICKER, sort=False)[col].shift(20).astype('float32')
    
    return df
