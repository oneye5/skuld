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
    # Make a copy to avoid SettingWithCopyWarning when we modify dtype
    df = df.copy()
    
    # Convert categorical columns to string to avoid memory issues in pivot
    # Categorical groupby creates cartesian product which uses huge memory
    if df[TICKER].dtype.name == 'category':
        df[TICKER] = df[TICKER].astype(str)
    if df[FEATURE].dtype.name == 'category':
        df[FEATURE] = df[FEATURE].astype(str)
    
    # Separate ticker data and macro data
    is_macro = df[FEATURE].str.startswith(MACRO_PREFIX)
    ticker_df = df.loc[~is_macro]  # Avoid copy by using loc
    macro_df = df.loc[is_macro]  # Avoid copy by using loc

    # Get Close timestamps as anchors (only for non-empty tickers)
    close_rows = ticker_df.loc[
        (ticker_df[FEATURE] == CLOSE) & (ticker_df[TICKER] != ""),
        [TIMESTAMP, TICKER]
    ].drop_duplicates()

    if close_rows.empty:
        return pd.DataFrame(columns=[TIMESTAMP, TICKER])

    # Pivot ticker data to wide format
    ticker_wide = _pivot_ticker_data(ticker_df, close_rows)

    # Merge macro data using vectorized approach
    if not macro_df.empty:
        ticker_wide = _merge_macro_data_vectorized(ticker_wide, macro_df)
    
    # Convert numeric columns to float32 to save memory (50% reduction)
    for col in ticker_wide.columns:
        if ticker_wide[col].dtype == 'float64':
            ticker_wide[col] = ticker_wide[col].astype('float32')

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

    # Filter to only Close timestamps using more efficient merge
    pivoted = pivoted.merge(close_rows, on=[TICKER, TIMESTAMP], how='inner')
    
    return pivoted


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
