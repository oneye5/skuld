"""Profile a single ranking window to identify per-window bottlenecks.

This script profiles each step within run_single_ranking_window to identify
where time is being spent during the window processing phase.

Usage:
    uv run python scripts/profile_single_window.py
    uv run python scripts/profile_single_window.py --compare-implementations
"""

import time
import sys
import gc
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER
from config.settings import MS_PER_DAY


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
        print(f"    {self.name}: {elapsed:.3f}s")


def profile_single_window():
    """Profile each step of a single ranking window."""
    results = {}
    
    print("=" * 60)
    print("SINGLE WINDOW PROFILING")
    print("=" * 60)
    
    # Load data with features (should be cached)
    print("\n[1] LOADING DATA (with feature cache)")
    with Timer("load_data_with_features", results):
        from pipeline.ranking_pipeline import prepare_wide_data_with_features
        wide_df = prepare_wide_data_with_features(use_cache=True)
    print(f"    Shape: {wide_df.shape}")
    
    # Setup window parameters
    forward_return_days = 365
    lookahead_ms = forward_return_days * MS_PER_DAY
    
    data_max_ts = wide_df[TIMESTAMP].max()
    data_min_ts = wide_df[TIMESTAMP].min()
    
    # Use a window that actually has data
    # Test period: last 0.5 years from max_ts - lookahead
    test_end_ts = int(data_max_ts - lookahead_ms)  # Ensure we have future data for returns
    train_end_ts = int(test_end_ts - 0.5 * 365 * 86400 * 1000)  # 0.5 years test
    
    print(f"\n[2] WINDOW SETUP")
    print(f"    Data range: {data_min_ts} - {data_max_ts}")
    print(f"    Train end: {train_end_ts}")
    print(f"    Test end: {test_end_ts}")
    
    # Profile each step of window processing
    print("\n[3] PROFILING WINDOW STEPS")
    
    # Step 1: Slice data
    with Timer("slice_data", results):
        buffer_end = test_end_ts + lookahead_ms
        wide_slice = wide_df[wide_df[TIMESTAMP] < buffer_end].copy()
    print(f"       Slice shape: {wide_slice.shape}")
    
    # Step 2: Split data
    with Timer("split_data", results):
        from core.splitter import split_by_timestamp
        split = split_by_timestamp(wide_slice, train_end_ts, test_end_ts)
    print(f"       Train: {len(split.train):,}, Test: {len(split.test):,}")
    
    # Step 3: Compute forward returns (vectorized)
    with Timer("forward_returns_vectorized", results):
        from core.target_builder import compute_forward_returns
        train_with_returns = compute_forward_returns(
            split.train,
            lookahead_days=forward_return_days,
            return_type="simple",
            winsorize_limits=(-0.5, 0.5),
            drop_na=True,
            price_lookup_df=wide_slice,
        )
        test_with_returns = compute_forward_returns(
            split.test,
            lookahead_days=forward_return_days,
            return_type="simple",
            winsorize_limits=(-0.5, 0.5),
            drop_na=True,
            price_lookup_df=wide_slice,
        )
    print(f"       Train with returns: {len(train_with_returns):,}")
    print(f"       Test with returns: {len(test_with_returns):,}")
    
    del wide_slice
    gc.collect()
    
    # Step 4: Filter min stocks
    with Timer("filter_min_stocks", results):
        from learner.ranking import filter_min_stocks_per_timestamp
        train_filtered = filter_min_stocks_per_timestamp(train_with_returns, 10, TIMESTAMP)
        test_filtered = filter_min_stocks_per_timestamp(test_with_returns, 10, TIMESTAMP)
    
    # Step 5: Preprocess
    with Timer("preprocess", results):
        from core.preprocessor import preprocess_data
        train_processed = preprocess_data(train_filtered, add_missing_flags=False)
        test_processed = preprocess_data(test_filtered, add_missing_flags=False)
    
    # Step 6: Get feature columns
    with Timer("get_feature_columns", results):
        from pipeline.ranking_pipeline import get_feature_columns_for_ranking
        train_feature_cols = set(get_feature_columns_for_ranking(train_processed))
        test_feature_cols = set(get_feature_columns_for_ranking(test_processed))
        feature_cols = sorted(train_feature_cols & test_feature_cols)
    print(f"       Features: {len(feature_cols)}")
    
    # Step 7: Scale
    with Timer("scale", results):
        from core.scaler import fit_scaler, transform_data
        scaler = fit_scaler(train_processed)
        train_scaled = transform_data(train_processed, scaler)
        test_scaled = transform_data(test_processed, scaler)
    
    # Step 8: Clip extreme values
    with Timer("clip_extreme", results):
        from core.preprocessor import clip_extreme_values
        train_scaled = clip_extreme_values(train_scaled)
        test_scaled = clip_extreme_values(test_scaled)
    
    # Step 9: Prepare ranking data
    with Timer("prepare_ranking_data", results):
        from learner.ranking import prepare_ranking_data
        from core.target_builder import FORWARD_RETURN
        X_train, y_train, groups_train = prepare_ranking_data(
            train_scaled, feature_cols, FORWARD_RETURN, TIMESTAMP
        )
        X_test, y_test, groups_test = prepare_ranking_data(
            test_scaled, feature_cols, FORWARD_RETURN, TIMESTAMP
        )
    print(f"       X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Step 10: Train model
    with Timer("train_model", results):
        from learner.ranking import LightGBMRankerWrapper, RankerConfig
        config = RankerConfig(n_estimators=100)
        ranker = LightGBMRankerWrapper(config)
        ranker.fit(X_train, y_train, groups_train)
    
    # Step 11: Predict
    with Timer("predict", results):
        predictions = ranker.predict(X_test)
    
    # Summary
    print("\n" + "=" * 60)
    print("WINDOW PROFILING SUMMARY")
    print("=" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    total = sum(v for k, v in results.items() if k != "load_data_with_features")
    
    print(f"\n{'Stage':<35} {'Time (s)':>10} {'%':>8}")
    print("-" * 55)
    for name, elapsed in sorted_results:
        if name == "load_data_with_features":
            continue
        pct = (elapsed / total) * 100 if total > 0 else 0
        print(f"{name:<35} {elapsed:>10.3f} {pct:>7.1f}%")
    print("-" * 55)
    print(f"{'TOTAL (excl. data load)':<35} {total:>10.3f}")
    print(f"{'Data loading (cached)':<35} {results['load_data_with_features']:>10.3f}")
    
    return results


def compare_forward_returns_implementations():
    """Compare vectorized vs loop-based forward returns."""
    print("\n" + "=" * 60)
    print("COMPARING FORWARD RETURN IMPLEMENTATIONS")
    print("=" * 60)
    
    from pipeline.ranking_pipeline import prepare_wide_data
    from core.target_builder import compute_forward_returns, compute_forward_returns_loop
    
    # Load data
    print("\nLoading data...")
    wide_df = prepare_wide_data(use_cache=True)
    
    # Take a subset for fair comparison
    print("Taking subset for comparison...")
    sample_df = wide_df.sample(min(100000, len(wide_df)), random_state=42).copy()
    print(f"Sample size: {len(sample_df):,} rows")
    
    # Time vectorized
    print("\nTiming vectorized implementation...")
    start = time.perf_counter()
    result_vec = compute_forward_returns(
        sample_df.copy(),
        lookahead_days=20,
        drop_na=True,
    )
    vec_time = time.perf_counter() - start
    print(f"  Vectorized: {vec_time:.3f}s, {len(result_vec):,} rows")
    
    # Time loop-based
    print("\nTiming loop-based implementation...")
    start = time.perf_counter()
    result_loop = compute_forward_returns_loop(
        sample_df.copy(),
        lookahead_days=20,
        drop_na=True,
    )
    loop_time = time.perf_counter() - start
    print(f"  Loop-based: {loop_time:.3f}s, {len(result_loop):,} rows")
    
    # Compare
    print(f"\nSpeedup: {loop_time/vec_time:.1f}x")
    
    return {"vectorized": vec_time, "loop": loop_time}


def compare_technical_feature_implementations():
    """Compare vectorized vs loop-based technical features."""
    print("\n" + "=" * 60)
    print("COMPARING TECHNICAL FEATURE IMPLEMENTATIONS")
    print("=" * 60)
    
    from core.data_cache import load_cached_wide_data
    
    # Import both implementations
    import features.technical as tech_module
    
    # Load data without features
    print("\nLoading raw wide data (without features)...")
    wide_df = load_cached_wide_data()
    
    # Take subset
    print("Preparing test data...")
    sample_df = wide_df.copy()
    print(f"Data size: {len(sample_df):,} rows, {sample_df[TICKER].nunique()} tickers")
    
    # Time vectorized
    print("\nTiming vectorized implementation...")
    tech_module.USE_VECTORIZED = True
    start = time.perf_counter()
    result_vec = tech_module.add_technical_features(sample_df.copy())
    vec_time = time.perf_counter() - start
    print(f"  Vectorized: {vec_time:.2f}s")
    
    # Time loop-based
    print("\nTiming loop-based implementation...")
    tech_module.USE_VECTORIZED = False
    start = time.perf_counter()
    result_loop = tech_module.add_technical_features(sample_df.copy())
    loop_time = time.perf_counter() - start
    print(f"  Loop-based: {loop_time:.2f}s")
    
    # Reset to default
    tech_module.USE_VECTORIZED = True
    
    # Compare
    print(f"\nSpeedup: {loop_time/vec_time:.1f}x")
    
    # Verify outputs match (approximately)
    print("\nVerifying outputs match...")
    common_cols = set(result_vec.columns) & set(result_loop.columns)
    numeric_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(result_vec[c])]
    
    mismatches = []
    for col in numeric_cols[:10]:  # Check first 10
        vec_vals = result_vec[col].dropna()
        loop_vals = result_loop[col].dropna()
        if len(vec_vals) > 0 and len(loop_vals) > 0:
            # Compare means
            vec_mean = vec_vals.mean()
            loop_mean = loop_vals.mean()
            if abs(vec_mean - loop_mean) > 0.01 * abs(loop_mean + 1e-10):
                mismatches.append(col)
    
    if mismatches:
        print(f"  WARNING: Potential mismatches in: {mismatches}")
    else:
        print(f"  ✓ Outputs match (checked {len(numeric_cols[:10])} columns)")
    
    return {"vectorized": vec_time, "loop": loop_time}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-implementations", action="store_true",
                       help="Compare vectorized vs loop-based implementations")
    args = parser.parse_args()
    
    if args.compare_implementations:
        compare_forward_returns_implementations()
        compare_technical_feature_implementations()
    else:
        profile_single_window()
