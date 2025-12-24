#!/usr/bin/env python3
"""Debug script for inspecting data at various pipeline stages.

Usage:
    uv run scripts/debug_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.columns import TIMESTAMP, TICKER, CLOSE
from core.data_loader import load_long_data
from pipeline.single_window import prepare_wide_data


def main():
    """Debug data loading and conversion."""
    print("Loading long format data...")
    long_df = load_long_data()
    
    print(f"\n--- Long Format ---")
    print(f"Shape: {long_df.shape}")
    print(f"Columns: {list(long_df.columns)}")
    print(f"Memory: {long_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"\nSample:")
    print(long_df.head(10))
    
    print("\n" + "=" * 60)
    print("Converting to wide format...")
    wide_df = prepare_wide_data(long_df)
    
    print(f"\n--- Wide Format ---")
    print(f"Shape: {wide_df.shape}")
    print(f"Columns ({len(wide_df.columns)}): {list(wide_df.columns)[:20]}...")
    print(f"Memory: {wide_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    print(f"\nTimestamp range:")
    from datetime import datetime
    min_ts = wide_df[TIMESTAMP].min()
    max_ts = wide_df[TIMESTAMP].max()
    print(f"  {datetime.utcfromtimestamp(min_ts/1000)} to {datetime.utcfromtimestamp(max_ts/1000)}")
    
    print(f"\nUnique tickers: {wide_df[TICKER].nunique()}")
    print(f"Sample tickers: {wide_df[TICKER].unique()[:10].tolist()}")
    
    print(f"\nSample data:")
    print(wide_df[[TIMESTAMP, TICKER, CLOSE]].head(10))


if __name__ == "__main__":
    main()
