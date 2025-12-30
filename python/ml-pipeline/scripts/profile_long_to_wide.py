#!/usr/bin/env python
"""Profile the long_to_wide conversion in detail."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import pandas as pd
from core.data_loader import load_long_data
from config.settings import YEAR_2000_MS
from config.columns import TIMESTAMP, TICKER, FEATURE, VALUE
from core.long_to_wide import clean_and_classify_tickers, add_macro_prefix, long_to_wide

def profile_step(name: str, func, *args, **kwargs):
    """Time a function call."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.2f}s")
    return result, elapsed

def main():
    """Profile each step of long_to_wide."""
    print("=" * 60)
    print("PROFILING LONG_TO_WIDE CONVERSION")
    print("=" * 60)
    
    total_start = time.perf_counter()
    
    # Step 1: Load data
    long_df, t1 = profile_step("1. load_long_data", load_long_data)
    print(f"   Shape: {long_df.shape}")
    
    # Step 2: Filter post-2000
    def filter_post_2000(df):
        return df[df[TIMESTAMP] >= YEAR_2000_MS].copy()
    
    long_df, t2 = profile_step("2. filter_post_2000", filter_post_2000, long_df)
    print(f"   Shape: {long_df.shape}")
    
    # Step 3: Clean tickers
    long_df, t3 = profile_step("3. clean_and_classify_tickers", clean_and_classify_tickers, long_df)
    print(f"   Shape: {long_df.shape}")
    
    # Step 4: Add macro prefix
    long_df, t4 = profile_step("4. add_macro_prefix", add_macro_prefix, long_df)
    print(f"   Shape: {long_df.shape}")
    
    # Step 5: long_to_wide (the main bottleneck)
    wide_df, t5 = profile_step("5. long_to_wide", long_to_wide, long_df)
    print(f"   Shape: {wide_df.shape}")
    
    total = time.perf_counter() - total_start
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time: {total:.2f}s")
    print()
    print("Breakdown:")
    times = [
        ("load_long_data", t1),
        ("filter_post_2000", t2),
        ("clean_and_classify_tickers", t3),
        ("add_macro_prefix", t4),
        ("long_to_wide", t5),
    ]
    for name, t in sorted(times, key=lambda x: -x[1]):
        pct = t / total * 100
        print(f"  {name:<30} {t:>6.2f}s ({pct:5.1f}%)")
    
    # Check if caching would help
    print()
    print("=" * 60)
    print("CACHING ANALYSIS")
    print("=" * 60)
    
    # Try saving as parquet
    cache_path = Path("output/cache/wide_df.parquet")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    start = time.perf_counter()
    wide_df.to_parquet(cache_path, index=False, compression='snappy')
    save_time = time.perf_counter() - start
    print(f"Save to parquet: {save_time:.2f}s")
    
    # Check file size
    size_mb = cache_path.stat().st_size / 1024 / 1024
    print(f"File size: {size_mb:.1f} MB")
    
    # Try loading from parquet
    start = time.perf_counter()
    loaded = pd.read_parquet(cache_path)
    load_time = time.perf_counter() - start
    print(f"Load from parquet: {load_time:.2f}s")
    
    # Verify
    print(f"Loaded shape: {loaded.shape}")
    
    print()
    print(f"*** POTENTIAL SPEEDUP: {total / load_time:.1f}x by using parquet cache ***")
    print(f"    (from {total:.1f}s to {load_time:.1f}s)")


if __name__ == "__main__":
    main()
