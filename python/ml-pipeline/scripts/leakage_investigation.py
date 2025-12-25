"""
Leakage Investigation Script

This script investigates potential sources of data leakage that could explain
why the old nzx-predictor achieved ~1+ Sharpe vs the current pipeline's ~0.086.

Key areas to investigate:
1. Scaler fitting on train+test (known issue in nzx-predictor)
2. Feature engineering using future data
3. Label leakage (target info in features)
4. Time-based leakage (using future timestamps/features)
5. Train/test overlap or contamination
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

from config.columns import TIMESTAMP, TICKER, CLOSE, TARGET
from config.settings import LOOKAHEAD_DAYS, MS_PER_DAY, GAIN_THRESHOLD_PCT
from core.data_loader import load_long_data
from core.long_to_wide import long_to_wide, add_macro_prefix, clean_and_classify_tickers
from core.labeler import create_labels
from core.splitter import split_by_timestamp


def ts_to_date(ts: int) -> str:
    """Convert timestamp to readable date."""
    return datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")


def investigate_label_leakage(wide_df: pd.DataFrame) -> dict:
    """
    Check if there's any correlation between target and features that 
    shouldn't exist (potential label leakage).
    """
    print("\n" + "="*60)
    print("1. INVESTIGATING LABEL LEAKAGE")
    print("="*60)
    
    # Create labels
    labeled = create_labels(wide_df.copy(), LOOKAHEAD_DAYS, GAIN_THRESHOLD_PCT)
    
    if TARGET not in labeled.columns:
        return {"error": "No target column created"}
    
    # Get numeric feature columns
    excluded = [TIMESTAMP, TICKER, TARGET, CLOSE]
    feature_cols = [c for c in labeled.columns 
                   if c not in excluded 
                   and pd.api.types.is_numeric_dtype(labeled[c])
                   and labeled[c].std() > 0]
    
    # Check correlation of each feature with target
    correlations = []
    for col in feature_cols[:100]:  # Limit to 100 for speed
        corr = labeled[col].corr(labeled[TARGET])
        if not np.isnan(corr):
            correlations.append((col, abs(corr), corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 20 features by correlation with target:")
    for col, abs_corr, corr in correlations[:20]:
        print(f"  {col:50s}: {corr:+.4f}")
    
    # SUSPICIOUS: If any feature has very high correlation, it might be leaky
    suspicious = [c for c in correlations if c[1] > 0.3]
    if suspicious:
        print(f"\n⚠️  WARNING: {len(suspicious)} features have |correlation| > 0.3")
        print("   This could indicate label leakage!")
        for col, abs_corr, corr in suspicious:
            print(f"   - {col}: {corr:+.4f}")
    else:
        print("\n✓ No features with suspiciously high target correlation (>0.3)")
    
    return {
        "num_features_checked": len(correlations),
        "suspicious_features": suspicious,
        "max_correlation": correlations[0] if correlations else None
    }


def investigate_temporal_leakage(wide_df: pd.DataFrame) -> dict:
    """
    Check if feature values "look into the future" by examining 
    whether test set features contain information from after the train period.
    
    Also check if there's any overlap in timestamps between train and test.
    """
    print("\n" + "="*60)
    print("2. INVESTIGATING TEMPORAL LEAKAGE")
    print("="*60)
    
    # Use a simple split
    max_ts = wide_df[TIMESTAMP].max()
    test_period_ms = 30 * MS_PER_DAY  # 30 days test
    train_end_ts = max_ts - LOOKAHEAD_DAYS * MS_PER_DAY - test_period_ms
    test_end_ts = max_ts - LOOKAHEAD_DAYS * MS_PER_DAY
    
    split = split_by_timestamp(wide_df, train_end_ts, test_end_ts)
    
    print(f"\nSplit details:")
    print(f"  Train: {ts_to_date(split.train_start_ts)} to {ts_to_date(split.train_end_ts)}")
    print(f"  Test:  {ts_to_date(split.test_start_ts)} to {ts_to_date(split.test_end_ts)}")
    print(f"  Train samples: {len(split.train):,}")
    print(f"  Test samples:  {len(split.test):,}")
    
    # Check for timestamp overlap
    train_timestamps = set(split.train[TIMESTAMP])
    test_timestamps = set(split.test[TIMESTAMP])
    overlap = train_timestamps & test_timestamps
    
    if overlap:
        print(f"\n⚠️  WARNING: {len(overlap)} overlapping timestamps!")
        sample_overlap = list(overlap)[:5]
        for ts in sample_overlap:
            print(f"   - {ts_to_date(ts)}")
    else:
        print("\n✓ No timestamp overlap between train and test")
    
    # Check if same ticker appears in train and test (expected, not leakage by itself)
    train_tickers = set(split.train[TICKER])
    test_tickers = set(split.test[TICKER])
    common_tickers = train_tickers & test_tickers
    
    print(f"\nTicker overlap (expected):")
    print(f"  Train tickers: {len(train_tickers)}")
    print(f"  Test tickers:  {len(test_tickers)}")
    print(f"  Common:        {len(common_tickers)}")
    
    return {
        "timestamp_overlap": len(overlap),
        "common_tickers": len(common_tickers),
        "train_samples": len(split.train),
        "test_samples": len(split.test)
    }


def investigate_scaler_leakage(wide_df: pd.DataFrame) -> dict:
    """
    Demonstrate the difference between fitting scaler on train-only vs train+test.
    
    This was a known issue in nzx-predictor: fitting on combined data means
    test set statistics influence the scaler, which is leakage.
    """
    print("\n" + "="*60)
    print("3. INVESTIGATING SCALER LEAKAGE")
    print("="*60)
    
    from sklearn.preprocessing import RobustScaler
    
    # Use a simple split
    max_ts = wide_df[TIMESTAMP].max()
    test_period_ms = 30 * MS_PER_DAY
    train_end_ts = max_ts - LOOKAHEAD_DAYS * MS_PER_DAY - test_period_ms
    test_end_ts = max_ts - LOOKAHEAD_DAYS * MS_PER_DAY
    
    split = split_by_timestamp(wide_df, train_end_ts, test_end_ts)
    
    # Get numeric columns
    excluded = [TIMESTAMP, TICKER]
    numeric_cols = [c for c in split.train.columns 
                   if c not in excluded 
                   and pd.api.types.is_numeric_dtype(split.train[c])
                   and split.train[c].std() > 0][:50]  # Limit for speed
    
    if not numeric_cols:
        return {"error": "No numeric columns found"}
    
    # Method 1: Fit on train only (correct)
    scaler_train = RobustScaler()
    scaler_train.fit(split.train[numeric_cols])
    train_scaled_correct = scaler_train.transform(split.train[numeric_cols])
    test_scaled_correct = scaler_train.transform(split.test[numeric_cols])
    
    # Method 2: Fit on combined (leaky)
    combined = pd.concat([split.train[numeric_cols], split.test[numeric_cols]])
    scaler_combined = RobustScaler()
    scaler_combined.fit(combined)
    train_scaled_leaky = scaler_combined.transform(split.train[numeric_cols])
    test_scaled_leaky = scaler_combined.transform(split.test[numeric_cols])
    
    # Compare differences
    train_diff = np.abs(train_scaled_correct - train_scaled_leaky).mean()
    test_diff = np.abs(test_scaled_correct - test_scaled_leaky).mean()
    
    print(f"\nScaler comparison (train-only vs combined):")
    print(f"  Mean abs difference in train data: {train_diff:.6f}")
    print(f"  Mean abs difference in test data:  {test_diff:.6f}")
    
    # Compare scaler statistics
    print(f"\nScaler center comparison (first 5 features):")
    for i, col in enumerate(numeric_cols[:5]):
        center_train = scaler_train.center_[i]
        center_combined = scaler_combined.center_[i]
        diff_pct = abs(center_train - center_combined) / (abs(center_train) + 1e-6) * 100
        print(f"  {col[:40]:40s}: train={center_train:10.2f}, combined={center_combined:10.2f}, diff={diff_pct:.2f}%")
    
    if test_diff > 0.01:
        print(f"\n⚠️  WARNING: Scaler leakage causes {test_diff:.4f} average difference in test data")
        print("   Fitting on combined data is a source of leakage!")
    else:
        print("\n✓ Scaler leakage effect is minimal for this data")
    
    return {
        "train_diff": train_diff,
        "test_diff": test_diff,
        "num_features": len(numeric_cols)
    }


def investigate_labeling_process(wide_df: pd.DataFrame) -> dict:
    """
    Check the labeling process for potential issues:
    1. Are labels computed using only past data?
    2. Is there any future information leaking into the label?
    """
    print("\n" + "="*60)
    print("4. INVESTIGATING LABELING PROCESS")
    print("="*60)
    
    # Create labels and examine the process
    sample_ticker = wide_df[TICKER].value_counts().index[0]
    ticker_data = wide_df[wide_df[TICKER] == sample_ticker].sort_values(TIMESTAMP)
    
    print(f"\nExamining ticker: {sample_ticker}")
    print(f"  Total samples: {len(ticker_data)}")
    print(f"  Date range: {ts_to_date(ticker_data[TIMESTAMP].min())} to {ts_to_date(ticker_data[TIMESTAMP].max())}")
    
    # Create labels for this ticker
    labeled = create_labels(ticker_data.copy(), LOOKAHEAD_DAYS, GAIN_THRESHOLD_PCT)
    
    if TARGET not in labeled.columns or len(labeled) == 0:
        return {"error": "No labels created"}
    
    print(f"  Labeled samples: {len(labeled)}")
    print(f"  Label distribution: {dict(labeled[TARGET].value_counts())}")
    
    # Check: Does the last labeled row have enough future data?
    last_labeled_ts = labeled[TIMESTAMP].max()
    last_data_ts = ticker_data[TIMESTAMP].max()
    lookahead_needed = LOOKAHEAD_DAYS * MS_PER_DAY
    
    actual_buffer = last_data_ts - last_labeled_ts
    
    print(f"\n  Last labeled timestamp: {ts_to_date(last_labeled_ts)}")
    print(f"  Last data timestamp:    {ts_to_date(last_data_ts)}")
    print(f"  Buffer available:       {actual_buffer / MS_PER_DAY:.0f} days")
    print(f"  Buffer needed:          {LOOKAHEAD_DAYS} days")
    
    if actual_buffer < lookahead_needed:
        print(f"\n⚠️  WARNING: Insufficient future buffer ({actual_buffer/MS_PER_DAY:.0f} < {LOOKAHEAD_DAYS} days)")
    else:
        print("\n✓ Sufficient future buffer for labeling")
    
    # Check a sample label calculation
    mid_idx = len(labeled) // 2
    sample_row = labeled.iloc[mid_idx]
    sample_ts = sample_row[TIMESTAMP]
    sample_close = sample_row[CLOSE]
    sample_target = sample_row[TARGET]
    
    target_ts = sample_ts + lookahead_needed
    future_data = ticker_data[ticker_data[TIMESTAMP] >= target_ts].iloc[0] if len(ticker_data[ticker_data[TIMESTAMP] >= target_ts]) > 0 else None
    
    if future_data is not None:
        future_close = future_data[CLOSE]
        expected_pct_change = (future_close - sample_close) / sample_close * 100
        expected_target = 1 if expected_pct_change >= GAIN_THRESHOLD_PCT else 0
        
        print(f"\nSample label verification:")
        print(f"  Current date:   {ts_to_date(sample_ts)}")
        print(f"  Current price:  ${sample_close:.2f}")
        print(f"  Future date:    {ts_to_date(int(future_data[TIMESTAMP]))}")
        print(f"  Future price:   ${future_close:.2f}")
        print(f"  % Change:       {expected_pct_change:.2f}%")
        print(f"  Threshold:      {GAIN_THRESHOLD_PCT}%")
        print(f"  Expected label: {expected_target}")
        print(f"  Actual label:   {int(sample_target)}")
        
        if expected_target != int(sample_target):
            print("\n⚠️  WARNING: Label mismatch detected!")
        else:
            print("\n✓ Label calculation appears correct")
    
    return {
        "labeled_samples": len(labeled),
        "buffer_days": actual_buffer / MS_PER_DAY
    }


def investigate_feature_timing(wide_df: pd.DataFrame) -> dict:
    """
    Check if features might contain future-looking information.
    For example, if a 'moving average' is calculated using future prices.
    """
    print("\n" + "="*60)
    print("5. INVESTIGATING FEATURE TIMING")
    print("="*60)
    
    # This is harder to detect automatically - we'll flag suspicious column names
    feature_cols = [c for c in wide_df.columns if c not in [TIMESTAMP, TICKER]]
    
    suspicious_patterns = [
        'future', 'forward', 'next', 'tomorrow', 'target', 'label',
        'return', 'gain', 'profit', 'loss'
    ]
    
    suspicious_cols = []
    for col in feature_cols:
        col_lower = col.lower()
        for pattern in suspicious_patterns:
            if pattern in col_lower:
                suspicious_cols.append((col, pattern))
                break
    
    if suspicious_cols:
        print(f"\n⚠️  Found {len(suspicious_cols)} potentially suspicious column names:")
        for col, pattern in suspicious_cols[:20]:
            print(f"   - {col} (contains '{pattern}')")
    else:
        print("\n✓ No obviously suspicious column names detected")
    
    print(f"\nTotal feature columns: {len(feature_cols)}")
    print(f"Sample column names:")
    for col in feature_cols[:20]:
        print(f"  - {col}")
    
    return {
        "total_features": len(feature_cols),
        "suspicious_cols": [c[0] for c in suspicious_cols]
    }


def investigate_target_class_balance(wide_df: pd.DataFrame) -> dict:
    """
    Check the target class balance across different time periods.
    Large imbalances or inconsistencies could indicate issues.
    """
    print("\n" + "="*60)
    print("6. INVESTIGATING TARGET CLASS BALANCE")
    print("="*60)
    
    labeled = create_labels(wide_df.copy(), LOOKAHEAD_DAYS, GAIN_THRESHOLD_PCT)
    
    if TARGET not in labeled.columns or len(labeled) == 0:
        return {"error": "No labels created"}
    
    overall_positive_rate = labeled[TARGET].mean()
    print(f"\nOverall positive rate: {overall_positive_rate:.2%}")
    print(f"Total samples: {len(labeled):,}")
    
    # Check by year
    labeled['year'] = pd.to_datetime(labeled[TIMESTAMP] / 1000, unit='s').dt.year
    yearly_rates = labeled.groupby('year')[TARGET].agg(['mean', 'count'])
    
    print(f"\nPositive rate by year:")
    for year, row in yearly_rates.iterrows():
        print(f"  {year}: {row['mean']:.2%} (n={row['count']:,})")
    
    # Check for extreme variations
    rate_std = yearly_rates['mean'].std()
    rate_range = yearly_rates['mean'].max() - yearly_rates['mean'].min()
    
    if rate_range > 0.3:
        print(f"\n⚠️  WARNING: Large variation in positive rate across years (range={rate_range:.2%})")
        print("   This could indicate regime changes or data issues")
    else:
        print(f"\n✓ Positive rate reasonably consistent across years (range={rate_range:.2%})")
    
    return {
        "overall_positive_rate": overall_positive_rate,
        "rate_std": rate_std,
        "rate_range": rate_range
    }


def main():
    print("="*60)
    print("LEAKAGE INVESTIGATION FOR SKULD ML PIPELINE")
    print("="*60)
    print(f"\nConfig:")
    print(f"  LOOKAHEAD_DAYS: {LOOKAHEAD_DAYS}")
    print(f"  GAIN_THRESHOLD_PCT: {GAIN_THRESHOLD_PCT}%")
    
    # Load data
    print("\nLoading data...")
    long_df = load_long_data()
    print(f"Loaded {len(long_df):,} rows")
    
    # Filter and prepare
    from config.settings import YEAR_2000_MS
    long_df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS]
    long_df = clean_and_classify_tickers(long_df)
    long_df = add_macro_prefix(long_df)
    wide_df = long_to_wide(long_df)
    print(f"Wide format: {len(wide_df):,} rows, {len(wide_df.columns)} columns")
    
    # Run investigations
    results = {}
    
    results['label_leakage'] = investigate_label_leakage(wide_df)
    results['temporal_leakage'] = investigate_temporal_leakage(wide_df)
    results['scaler_leakage'] = investigate_scaler_leakage(wide_df)
    results['labeling_process'] = investigate_labeling_process(wide_df)
    results['feature_timing'] = investigate_feature_timing(wide_df)
    results['class_balance'] = investigate_target_class_balance(wide_df)
    
    # Summary
    print("\n" + "="*60)
    print("INVESTIGATION SUMMARY")
    print("="*60)
    
    issues_found = []
    
    if results['label_leakage'].get('suspicious_features'):
        issues_found.append("High feature-target correlations detected")
    
    if results['temporal_leakage'].get('timestamp_overlap', 0) > 0:
        issues_found.append("Timestamp overlap between train/test")
    
    if results['scaler_leakage'].get('test_diff', 0) > 0.01:
        issues_found.append("Significant scaler leakage effect")
    
    if results['feature_timing'].get('suspicious_cols'):
        issues_found.append(f"Suspicious column names: {results['feature_timing']['suspicious_cols'][:5]}")
    
    if results['class_balance'].get('rate_range', 0) > 0.3:
        issues_found.append("Large variation in positive rate across years")
    
    if issues_found:
        print("\n⚠️  POTENTIAL ISSUES FOUND:")
        for issue in issues_found:
            print(f"   - {issue}")
    else:
        print("\n✓ No obvious leakage issues detected in this pipeline")
    
    print("\n" + "-"*60)
    print("HYPOTHESIS: If nzx-predictor had a higher Sharpe (1+), the likely")
    print("leakage sources to investigate in the OLD codebase are:")
    print("  1. Scaler fitted on train+test combined (known issue)")
    print("  2. Features computed using all data (not respecting time splits)")
    print("  3. Rolling statistics using future data")
    print("  4. Label calculation issues")
    print("-"*60)
    
    return results


if __name__ == "__main__":
    results = main()
