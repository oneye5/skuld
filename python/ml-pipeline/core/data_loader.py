"""Data loader module for reading long format CSV data."""

from pathlib import Path
import pandas as pd

from config.columns import TIMESTAMP, TICKER, FEATURE, VALUE
from config.paths import DATA_LONG_CSV


def load_long_data(file_path: Path | None = None) -> pd.DataFrame:
    """Load the long format CSV data with memory-efficient dtypes.
    
    Args:
        file_path: Path to CSV file. Uses DATA_LONG_CSV if not provided.
    
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
        na_values=["", "null"],
        keep_default_na=True,
    )
    
    # Fill empty tickers with empty string for consistent handling
    df[TICKER] = df[TICKER].fillna("")
    
    return df
