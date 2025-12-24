#!/usr/bin/env python3
"""Analyze column sparsity to determine optimal threshold."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import load_long_data
from core.long_to_wide import add_macro_prefix, long_to_wide
from config.settings import YEAR_2000_MS
from config.columns import TIMESTAMP


def main():
    print("Loading data...")
    long_df = load_long_data()
    
    df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS].copy()
    df = add_macro_prefix(df)
    
    print("Converting to wide format (no sparse filtering)...")
    wide_df = long_to_wide(df)
    
    print(f"\nTotal rows: {len(wide_df):,}")
    print(f"Total columns: {len(wide_df.columns)}")
    
    # Check sparsity distribution
    missing = wide_df.isnull().mean().sort_values(ascending=False)
    
    print("\nColumn sparsity distribution:")
    for threshold in [0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05]:
        cols_above = (missing > threshold).sum()
        print(f"  >{threshold*100:.0f}% missing: {cols_above} columns")
    
    print("\nTop 30 sparsest columns:")
    print(missing.head(30).to_string())
    
    print("\nTop 30 least sparse columns (most complete):")
    print(missing.tail(30).to_string())


if __name__ == "__main__":
    main()
