"""One-hot encoding for ticker symbols.

Per nzx-predictor success: encoding tickers as features allows the model
to learn ticker-specific patterns. This is especially valuable when
tickers have different behaviors or risk profiles.
"""

import pandas as pd
from config.column_names import TICKER


def one_hot_encode_tickers(df: pd.DataFrame, prefix: str = "Ticker_") -> pd.DataFrame:
    """
    One-hot encode ticker column into binary feature columns.
    
    Args:
        df: DataFrame with TICKER column.
        prefix: Prefix for one-hot encoded column names.
    
    Returns:
        DataFrame with original columns plus one-hot encoded ticker columns.
    """
    if TICKER not in df.columns:
        return df
    
    # Use pd.get_dummies for efficient one-hot encoding
    dummies = pd.get_dummies(df[TICKER], prefix=prefix.rstrip("_"), dtype=int)
    
    # Concatenate all at once (avoids fragmentation)
    return pd.concat([df, dummies], axis=1)


def get_ticker_from_one_hot(
    df: pd.DataFrame,
    prefix: str = "Ticker_"
) -> pd.Series:
    """
    Recover ticker names from one-hot encoded columns.
    
    Args:
        df: DataFrame with one-hot encoded ticker columns.
        prefix: Prefix used during encoding.
    
    Returns:
        Series with ticker names.
    """
    ticker_cols = [c for c in df.columns if c.startswith(prefix)]
    if not ticker_cols:
        return pd.Series([""] * len(df), index=df.index)
    
    # Get the column with value 1 for each row
    ticker_df = df[ticker_cols]
    return (
        ticker_df.idxmax(axis=1)
        .str.replace(prefix, "", regex=False)
    )
