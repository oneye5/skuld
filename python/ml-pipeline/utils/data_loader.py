"""Data loader module for reading long format CSV data."""

from pathlib import Path
import pandas as pd

from config.column_names import TIMESTAMP, TICKER, FEATURE, VALUE
from config.file_paths import DATA_LONG_CSV


def load_long_data(file_path: Path | None = None) -> pd.DataFrame:
    """
    Load the long format CSV data with memory-efficient dtypes.
    
    Args:
        file_path: Path to CSV file. Uses default DATA_LONG_CSV if not provided.
    
    Returns:
        DataFrame with columns: timestamp, ticker, feature, value
    """
    path = file_path or DATA_LONG_CSV
    
    df = pd.read_csv(
        path,
        dtype={
            TIMESTAMP: "int64",
            TICKER: "str",
            FEATURE: "str",
            VALUE: "float32", 
        },
        na_values=[""],
        keep_default_na=True,
    )
    
    # Fill empty tickers with empty string for consistent handling
    df[TICKER] = df[TICKER].fillna("")
    
    # Optimize TICKER column memory usage (category is safe since it won't be modified)
    df[TICKER] = df[TICKER].astype('category')
    # Don't convert FEATURE to category yet - we'll modify it in add_macro_prefix
    
    return df
