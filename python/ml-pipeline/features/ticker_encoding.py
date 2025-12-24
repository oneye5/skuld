"""Ticker encoding features."""

import pandas as pd
import numpy as np

from config.columns import TICKER


def one_hot_encode_tickers(
    df: pd.DataFrame, 
    all_tickers: list[str] | None = None
) -> pd.DataFrame:
    """One-hot encode the ticker column with memory efficiency.
    
    Creates binary columns for each ticker (e.g., Ticker_ANZ.NZ).
    Uses uint8 dtype (1 byte) instead of int (8 bytes) for 8x memory savings.
    
    Args:
        df: DataFrame with ticker column.
        all_tickers: Optional list of all possible tickers. If provided,
            ensures consistent columns even if some tickers aren't in df.
    
    Returns:
        DataFrame with one-hot encoded ticker columns.
    """
    if TICKER not in df.columns:
        return df
    
    tickers = df[TICKER].values
    
    # Get unique tickers (use provided list or extract from data)
    if all_tickers is None:
        unique_tickers = sorted(df[TICKER].unique())
    else:
        unique_tickers = sorted(all_tickers)
    
    # Create ticker to index mapping
    ticker_to_idx = {t: i for i, t in enumerate(unique_tickers)}
    
    # Create one-hot matrix directly using numpy (memory efficient)
    n_rows = len(df)
    n_tickers = len(unique_tickers)
    
    # Use uint8 for massive memory savings (1 byte vs 8 bytes for int64)
    one_hot = np.zeros((n_rows, n_tickers), dtype=np.uint8)
    
    # Fill in the 1s
    for i, ticker in enumerate(tickers):
        if ticker in ticker_to_idx:
            one_hot[i, ticker_to_idx[ticker]] = 1
    
    # Create column names
    ticker_cols = [f"Ticker_{t}" for t in unique_tickers]
    
    # Build result DataFrame without the original ticker column
    result_cols = [c for c in df.columns if c != TICKER]
    result = df[result_cols].copy()
    
    # Add one-hot columns efficiently
    one_hot_df = pd.DataFrame(one_hot, columns=ticker_cols, index=df.index)
    result = pd.concat([result, one_hot_df], axis=1)
    
    return result


def encode_tickers_separately(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode train and test separately with consistent columns.
    
    This avoids concatenating large DataFrames which causes memory issues.
    
    Args:
        train_df: Training DataFrame with ticker column.
        test_df: Test DataFrame with ticker column.
    
    Returns:
        Tuple of (encoded_train, encoded_test) with consistent columns.
    """
    # Get all unique tickers from both sets
    all_tickers = sorted(set(train_df[TICKER].unique()) | set(test_df[TICKER].unique()))
    
    # Encode each separately with the full ticker list
    train_encoded = one_hot_encode_tickers(train_df, all_tickers)
    test_encoded = one_hot_encode_tickers(test_df, all_tickers)
    
    return train_encoded, test_encoded


def get_ticker_from_one_hot(df: pd.DataFrame) -> pd.Series:
    """Reconstruct ticker names from one-hot encoded columns.
    
    Args:
        df: DataFrame with one-hot encoded Ticker_* columns.
    
    Returns:
        Series with ticker names.
    """
    ticker_cols = [c for c in df.columns if c.startswith("Ticker_")]
    
    if not ticker_cols:
        raise ValueError("No one-hot encoded ticker columns found")
    
    return (
        df[ticker_cols]
        .idxmax(axis=1)
        .str.replace("Ticker_", "", regex=False)
    )
