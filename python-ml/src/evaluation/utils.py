"""Utility functions for loading and combining prediction files."""
import re
from pathlib import Path
from typing import Optional, List

import pandas as pd

from src.config.config import *
from src.preprocessing.pre_split_preprocessing import restore_ticker_column
from src.utils.csv_utils import load_csv, save_csv


def load_combined_predictions(directory: Path, pattern: str = "predictions*.csv") -> pd.DataFrame:
    """Load and combine multiple prediction CSV files.
    
    Handles:
    - Filtering numbered prediction files
    - Removing duplicate rows (by timestamp and ticker)
    - Restoring ticker column from one-hot encoding
    
    Args:
        directory: Path to directory containing prediction files.
        pattern: Glob pattern for matching files (default: predictions*.csv).
    
    Returns:
        pd.DataFrame: Combined predictions with duplicates removed, or empty DataFrame.
    
    Raises:
        FileNotFoundError: If directory doesn't exist.
    """
    if not Path(directory).exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(directory.glob(pattern))
    if not files:
        print(f"Warning: No files found matching '{pattern}' in {directory}")
        return pd.DataFrame()

    numbered_pattern = re.compile(r'predictions\d+\.csv$')
    numbered_files = [f for f in files if numbered_pattern.search(f.name)]

    if not numbered_files:
        print(f"Warning: No numbered prediction files found in {directory}")
        return pd.DataFrame()

    print(f"Found {len(numbered_files)} prediction files. Combining...")
    dfs = []

    for f in numbered_files:
        try:
            df = load_csv(str(f))
            if not df.empty:
                dfs.append(df)
                print(f"  ✓ Loaded: {f.name}")
        except Exception as e:
            print(f"  ✗ Failed to load {f.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = restore_ticker_column(combined_df)

    if TIMESTAMP_COL in combined_df.columns and TICKER_COL in combined_df.columns:
        initial_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(
            subset=[TIMESTAMP_COL, TICKER_COL],
            keep='last'
        )
        dropped = initial_len - len(combined_df)
        if dropped > 0:
            print(f"Dropped {dropped} duplicate rows.")

    return combined_df