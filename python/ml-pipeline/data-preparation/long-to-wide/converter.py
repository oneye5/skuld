"""Module for converting long format data to wide format."""

import pandas as pd
import numpy as np

from config.column_names import TIMESTAMP, TICKER, FEATURE, VALUE, CLOSE, MACRO_PREFIX


def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long format data to wide format.
    
    Uses 'Close' observations as timestamp anchors. For each ticker's Close 
    timestamp, all available features at or before that timestamp are joined.
    Macro features (MACRO_ prefix) are forward-filled up to each timestamp.
    
    Args:
        df: Long format DataFrame with timestamp, ticker, feature, value columns.
    
    Returns:
        Wide format DataFrame with timestamp, ticker, and feature columns.
    """
    # Separate ticker data and macro data
    is_macro = df[FEATURE].str.startswith(MACRO_PREFIX)
    ticker_df = df[~is_macro].copy()
    macro_df = df[is_macro].copy()
    
    # Get Close timestamps as anchors (only for non-empty tickers)
    close_rows = ticker_df[
        (ticker_df[FEATURE] == CLOSE) & (ticker_df[TICKER] != "")
    ][[TIMESTAMP, TICKER]].drop_duplicates()
    
    if close_rows.empty:
        return pd.DataFrame(columns=[TIMESTAMP, TICKER])
    
    # Pivot ticker data to wide format
    ticker_wide = _pivot_ticker_data(ticker_df, close_rows)
    
    # Merge macro data using vectorized approach
    if not macro_df.empty:
        ticker_wide = _merge_macro_data_vectorized(ticker_wide, macro_df)
    
    return ticker_wide.reset_index(drop=True)


def _pivot_ticker_data(ticker_df: pd.DataFrame, close_rows: pd.DataFrame) -> pd.DataFrame:
    """Pivot ticker data using Close timestamps as anchors."""
    # Filter ticker data to only include rows for tickers we have Close data for
    valid_tickers = close_rows[TICKER].unique()
    ticker_df = ticker_df[ticker_df[TICKER].isin(valid_tickers)]
    
    # Group by ticker and timestamp, take the last value if duplicates
    ticker_df = ticker_df.groupby([TICKER, TIMESTAMP, FEATURE])[VALUE].last().reset_index()
    
    # Pivot to wide format
    pivoted = ticker_df.pivot_table(
        index=[TICKER, TIMESTAMP],
        columns=FEATURE,
        values=VALUE,
        aggfunc='last'
    ).reset_index()
    
    # Filter to only Close timestamps
    close_set = set(zip(close_rows[TICKER], close_rows[TIMESTAMP]))
    mask = pivoted.apply(lambda row: (row[TICKER], row[TIMESTAMP]) in close_set, axis=1)
    
    return pivoted[mask].copy()


def _merge_macro_data_vectorized(ticker_wide: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """Merge macro data using vectorized pandas merge_asof."""
    if ticker_wide.empty or macro_df.empty:
        return ticker_wide
    
    # Pivot macro data to wide format
    macro_pivoted = macro_df.pivot_table(
        index=TIMESTAMP,
        columns=FEATURE,
        values=VALUE,
        aggfunc='last'
    ).reset_index()
    
    # Sort both dataframes by timestamp for merge_asof
    ticker_wide = ticker_wide.sort_values(TIMESTAMP)
    macro_pivoted = macro_pivoted.sort_values(TIMESTAMP)
    
    # Use merge_asof to find the most recent macro data for each ticker timestamp
    result = pd.merge_asof(
        ticker_wide,
        macro_pivoted,
        on=TIMESTAMP,
        direction='backward'
    )
    
    return result
