"""Profile the ranking pipeline to identify performance bottlenecks.

This script measures time spent in each major stage of data processing.
Run with: uv run python scripts/profile_pipeline.py
"""

import time
from pathlib import Path
import sys
import gc

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER


class Timer:
    """Simple timer context manager for profiling."""
    
    def __init__(self, name: str, results: dict):
        self.name = name
        self.results = results
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.results[self.name] = elapsed
        print(f"  {self.name}: {elapsed:.2f}s")


def profile_pipeline():
    """Profile each stage of the pipeline."""
    results = {}
    
    print("=" * 60)
    print("PIPELINE PROFILING")
    print("=" * 60)
    
    # 1. Data Loading
    print("\n[1] DATA LOADING")
    with Timer("load_long_data", results):
        from core.data_loader import load_long_data
        long_df = load_long_data()
    print(f"     Loaded {len(long_df):,} rows")
    
    # 2. Long to Wide Conversion
    print("\n[2] LONG TO WIDE CONVERSION")
    from config.settings import YEAR_2000_MS
    
    with Timer("filter_pre_2000", results):
        df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS].copy()
    print(f"     After filter: {len(df):,} rows")
    
    with Timer("clean_and_classify_tickers", results):
        from core.long_to_wide import clean_and_classify_tickers, add_macro_prefix
        df = clean_and_classify_tickers(df)
        df = add_macro_prefix(df)
    
    with Timer("long_to_wide", results):
        from core.long_to_wide import long_to_wide
        wide_df = long_to_wide(df)
    print(f"     Wide shape: {wide_df.shape}")
    del df
    gc.collect()
    
    with Timer("drop_sparse_columns", results):
        from core.preprocessor import drop_sparse_columns
        wide_df = drop_sparse_columns(wide_df, threshold=0.95)
    print(f"     After sparse drop: {wide_df.shape}")
    
    # 3. Feature Engineering - INDIVIDUAL TIMING
    print("\n[3] FEATURE ENGINEERING (detailed)")
    
    test_df = wide_df.copy()
    n_tickers = test_df[TICKER].nunique()
    print(f"     Testing on {len(test_df):,} rows, {n_tickers} tickers")
    
    with Timer("add_financial_ratios", results):
        from features.ratios import add_financial_ratios
        test_df = add_financial_ratios(test_df)
    
    with Timer("add_technical_features", results):
        from features.technical import add_technical_features
        test_df = add_technical_features(test_df)
    
    # 4. Preprocessing
    print("\n[4] PREPROCESSING")
    
    with Timer("preprocess_data", results):
        from core.preprocessor import preprocess_data
        test_df = preprocess_data(test_df, add_missing_flags=False)
    
    # 5. Scaling
    print("\n[5] SCALING")
    
    with Timer("fit_scaler", results):
        from core.scaler import fit_scaler
        scaler = fit_scaler(test_df)
    
    with Timer("transform_data", results):
        from core.scaler import transform_data
        test_df = transform_data(test_df, scaler)
    
    # 6. Target computation (simulate for profiling)
    print("\n[6] TARGET COMPUTATION (simulated)")
    
    with Timer("compute_forward_returns", results):
        from core.target_builder import compute_forward_returns
        # Use a small subset to estimate time
        sample_df = wide_df.sample(min(50000, len(wide_df))).copy()
        sample_df = compute_forward_returns(
            sample_df,
            lookahead_days=365,
            return_type="simple",
            drop_na=True,
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("PROFILING SUMMARY")
    print("=" * 60)
    
    # Sort by time descending
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    total = sum(results.values())
    
    print(f"\n{'Stage':<35} {'Time (s)':>10} {'%':>8}")
    print("-" * 55)
    for name, elapsed in sorted_results:
        pct = (elapsed / total) * 100
        print(f"{name:<35} {elapsed:>10.2f} {pct:>7.1f}%")
    print("-" * 55)
    print(f"{'TOTAL':<35} {total:>10.2f}")
    
    # Identify the big bottlenecks
    print("\n" + "=" * 60)
    print("TOP BOTTLENECKS (> 10% of time)")
    print("=" * 60)
    for name, elapsed in sorted_results:
        pct = (elapsed / total) * 100
        if pct > 10:
            print(f"  ⚠️  {name}: {elapsed:.2f}s ({pct:.1f}%)")
    
    return results


def profile_long_to_wide_detail():
    """Deep dive into long_to_wide - the likely bottleneck."""
    print("\n" + "=" * 60)
    print("LONG_TO_WIDE - DETAILED BREAKDOWN")
    print("=" * 60)
    
    from core.data_loader import load_long_data
    from config.settings import YEAR_2000_MS
    from core.long_to_wide import clean_and_classify_tickers, add_macro_prefix
    
    # Load and prepare data
    print("\nLoading data...")
    long_df = load_long_data()
    df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS].copy()
    df = clean_and_classify_tickers(df)
    df = add_macro_prefix(df)
    
    print(f"Data shape: {df.shape}")
    print(f"Unique tickers: {df[TICKER].nunique()}")
    print(f"Unique timestamps: {df[TIMESTAMP].nunique()}")
    
    # Profile each step of long_to_wide
    from config.columns import FEATURE, VALUE
    
    results = {}
    
    # Step 1: Group identification
    print("\nProfiling long_to_wide steps...")
    
    start = time.perf_counter()
    is_macro = df[TICKER].str.startswith('MACRO_')
    results["identify_macro"] = time.perf_counter() - start
    
    # Step 2: Split data
    start = time.perf_counter()
    macro_df = df[is_macro].copy()
    stock_df = df[~is_macro].copy()
    results["split_macro_stock"] = time.perf_counter() - start
    print(f"  Macro rows: {len(macro_df):,}, Stock rows: {len(stock_df):,}")
    
    # Step 3: Pivot stock data
    start = time.perf_counter()
    stock_wide = stock_df.pivot_table(
        index=[TIMESTAMP, TICKER],
        columns=FEATURE,
        values=VALUE,
        aggfunc='first'
    ).reset_index()
    results["pivot_stock"] = time.perf_counter() - start
    print(f"  Stock pivot shape: {stock_wide.shape}")
    
    # Step 4: Pivot macro data
    start = time.perf_counter()
    if len(macro_df) > 0:
        macro_wide = macro_df.pivot_table(
            index=TIMESTAMP,
            columns=TICKER,
            values=VALUE,
            aggfunc='first'
        ).reset_index()
    results["pivot_macro"] = time.perf_counter() - start
    print(f"  Macro pivot shape: {macro_wide.shape if len(macro_df) > 0 else 'N/A'}")
    
    # Step 5: Merge
    start = time.perf_counter()
    if len(macro_df) > 0:
        wide_df = stock_wide.merge(macro_wide, on=TIMESTAMP, how='left')
    else:
        wide_df = stock_wide
    results["merge"] = time.perf_counter() - start
    print(f"  Merged shape: {wide_df.shape}")
    
    # Summary
    print("\n" + "-" * 40)
    total = sum(results.values())
    for name, elapsed in sorted(results.items(), key=lambda x: x[1], reverse=True):
        pct = (elapsed / total) * 100
        print(f"  {name}: {elapsed:.2f}s ({pct:.1f}%)")
    print(f"  TOTAL: {total:.2f}s")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail", choices=["long_to_wide"], help="Detailed profiling")
    args = parser.parse_args()
    
    if args.detail == "long_to_wide":
        profile_long_to_wide_detail()
    else:
        profile_pipeline()
