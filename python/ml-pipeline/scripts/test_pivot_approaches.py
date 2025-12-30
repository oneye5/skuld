"""Test different pivot approaches for long_to_wide optimization."""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER, FEATURE, VALUE
from config.settings import YEAR_2000_MS


def load_test_data():
    """Load and prepare test data."""
    from core.data_loader import load_long_data
    from core.long_to_wide import clean_and_classify_tickers, add_macro_prefix
    
    long_df = load_long_data()
    df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS].copy()
    df = clean_and_classify_tickers(df)
    df = add_macro_prefix(df)
    
    # Filter to just non-macro for testing pivot
    df = df[~df[FEATURE].str.startswith('MACRO_')]
    return df


def approach_1_pivot_table(df):
    """Current approach: pivot_table."""
    start = time.perf_counter()
    result = df.pivot_table(
        index=[TIMESTAMP, TICKER],
        columns=FEATURE,
        values=VALUE,
        aggfunc='first'
    ).reset_index()
    elapsed = time.perf_counter() - start
    return result, elapsed


def approach_2_set_index_unstack(df):
    """Alternative: set_index + unstack (often faster)."""
    start = time.perf_counter()
    # Drop duplicates first to avoid ambiguous index
    df_dedup = df.drop_duplicates(subset=[TIMESTAMP, TICKER, FEATURE], keep='first')
    result = (df_dedup
              .set_index([TIMESTAMP, TICKER, FEATURE])[VALUE]
              .unstack(level=FEATURE)
              .reset_index())
    elapsed = time.perf_counter() - start
    return result, elapsed


def approach_3_groupby_first(df):
    """Alternative: groupby + first + unstack."""
    start = time.perf_counter()
    result = (df
              .groupby([TIMESTAMP, TICKER, FEATURE])[VALUE]
              .first()
              .unstack(level=FEATURE)
              .reset_index())
    elapsed = time.perf_counter() - start
    return result, elapsed


def approach_4_pivot(df):
    """Alternative: pivot (no aggregation)."""
    start = time.perf_counter()
    # Must handle duplicates first
    df_dedup = df.drop_duplicates(subset=[TIMESTAMP, TICKER, FEATURE], keep='first')
    result = df_dedup.pivot(
        index=[TIMESTAMP, TICKER],
        columns=FEATURE,
        values=VALUE
    ).reset_index()
    elapsed = time.perf_counter() - start
    return result, elapsed


def main():
    print("Loading test data...")
    df = load_test_data()
    print(f"Data shape: {df.shape}")
    print(f"Unique (timestamp, ticker) pairs: {df.groupby([TIMESTAMP, TICKER]).ngroups:,}")
    print(f"Unique features: {df[FEATURE].nunique()}")
    
    print("\n" + "=" * 60)
    print("COMPARING PIVOT APPROACHES")
    print("=" * 60)
    
    results = {}
    
    # Test each approach
    for name, func in [
        ("1. pivot_table (current)", approach_1_pivot_table),
        ("2. set_index + unstack", approach_2_set_index_unstack),
        ("3. groupby + first + unstack", approach_3_groupby_first),
        ("4. pivot (no agg)", approach_4_pivot),
    ]:
        print(f"\nTesting {name}...")
        try:
            result, elapsed = func(df.copy())
            results[name] = {
                "time": elapsed,
                "shape": result.shape,
                "memory_mb": result.memory_usage(deep=True).sum() / 1e6,
            }
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Shape: {result.shape}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"time": float('inf'), "error": str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    valid_results = [(k, v) for k, v in results.items() if "error" not in v]
    sorted_results = sorted(valid_results, key=lambda x: x[1]["time"])
    
    print(f"\n{'Approach':<35} {'Time (s)':>10} {'Shape':>20}")
    print("-" * 70)
    for name, r in sorted_results:
        print(f"{name:<35} {r['time']:>10.2f} {str(r['shape']):>20}")
    
    if sorted_results:
        best_name, best_result = sorted_results[0]
        current_time = results.get("1. pivot_table (current)", {}).get("time", 0)
        if current_time > 0:
            speedup = current_time / best_result["time"]
            print(f"\n✓ Best approach: {best_name}")
            print(f"  Speedup vs current: {speedup:.2f}x")


if __name__ == "__main__":
    main()
