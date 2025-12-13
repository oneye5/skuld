"""Data validation utilities to detect leakage, anomalies, and data quality issues."""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from src.config.config import TIMESTAMP_COL, LABEL_COL, TICKER_COL


def validate_time_series_integrity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ticker_col: str = TICKER_COL
) -> Dict[str, bool]:
    """
    Validate that train and test sets don't have temporal overlap (no leakage).
    
    Args:
        train_df: Training DataFrame with timestamp column.
        test_df: Test DataFrame with timestamp column.
        ticker_col: Column name for ticker (if one-hot encoded, check for pattern).
    
    Returns:
        Dictionary with validation results.
    """
    results = {}
    
    if TIMESTAMP_COL not in train_df.columns or TIMESTAMP_COL not in test_df.columns:
        results['timestamp_column_exists'] = False
        return results
    
    results['timestamp_column_exists'] = True
    
    # Check for temporal overlap
    train_max_ts = train_df[TIMESTAMP_COL].max()
    test_min_ts = test_df[TIMESTAMP_COL].min()
    test_max_ts = test_df[TIMESTAMP_COL].max()
    
    results['no_temporal_overlap'] = train_max_ts < test_min_ts
    
    if not results['no_temporal_overlap']:
        overlap_pct = (min(train_max_ts, test_max_ts) - test_min_ts) / (test_max_ts - test_min_ts)
        print(f"WARNING: Train-test temporal overlap detected! {overlap_pct*100:.1f}% of test set overlaps with training")
    
    results['train_time_range'] = (train_df[TIMESTAMP_COL].min(), train_max_ts)
    results['test_time_range'] = (test_min_ts, test_max_ts)
    
    return results


def check_data_quality(df: pd.DataFrame, name: str = "DataFrame") -> Dict[str, any]:
    """
    Comprehensive data quality checks.
    
    Args:
        df: DataFrame to validate.
        name: Name of the dataset for reporting.
    
    Returns:
        Dictionary with quality metrics.
    """
    results = {
        'name': name,
        'shape': df.shape,
        'empty': df.empty,
        'duplicates': df.duplicated().sum(),
    }
    
    # Check for NaN
    nan_counts = df.isna().sum()
    results['total_nans'] = nan_counts.sum()
    results['columns_with_nans'] = nan_counts[nan_counts > 0].to_dict() if nan_counts.sum() > 0 else {}
    
    # Check for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    inf_mask = np.isinf(numeric_df.values)
    results['total_infinites'] = inf_mask.sum()
    
    # Check for constant columns (no variance) - use vectorized nunique for speed
    nunique_counts = df.nunique()
    constant_cols = nunique_counts[nunique_counts <= 1].index.tolist()
    results['constant_columns'] = constant_cols
    
    # Check for label balance (if label column exists)
    if LABEL_COL in df.columns:
        label_counts = df[LABEL_COL].value_counts().to_dict()
        results['label_distribution'] = label_counts
        total = sum(label_counts.values())
        results['label_balance_ratio'] = max(label_counts.values()) / min(label_counts.values()) if min(label_counts.values()) > 0 else float('inf')
    
    # Data quality score
    quality_issues = [
        results['empty'],
        results['total_nans'] > 0,
        results['total_infinites'] > 0,
        len(constant_cols) > 0,
    ]
    results['quality_score'] = 1.0 - (sum(quality_issues) / len(quality_issues))
    
    return results


def detect_feature_drift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols_only: bool = True
) -> Dict[str, Dict]:
    """
    Detect feature drift between train and test sets.
    
    Compares statistical properties to detect dataset shift.
    
    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        numeric_cols_only: Only check numeric columns.
    
    Returns:
        Dictionary with drift metrics for each column.
    """
    drift_analysis = {}
    
    # Select columns to analyze
    cols = train_df.columns
    if numeric_cols_only:
        cols = train_df.select_dtypes(include=[np.number]).columns
    
    # Exclude special columns
    cols = [c for c in cols if c not in [TIMESTAMP_COL, LABEL_COL]]
    
    for col in cols:
        if col not in test_df.columns:
            continue
        
        train_vals = train_df[col].dropna()
        test_vals = test_df[col].dropna()
        
        if len(train_vals) == 0 or len(test_vals) == 0:
            continue
        
        drift_info = {
            'train_mean': train_vals.mean(),
            'test_mean': test_vals.mean(),
            'train_std': train_vals.std(),
            'test_std': test_vals.std(),
            'train_min': train_vals.min(),
            'test_min': test_vals.min(),
            'train_max': train_vals.max(),
            'test_max': test_vals.max(),
        }
        
        # Calculate relative drift
        if drift_info['train_mean'] != 0:
            drift_info['mean_drift_pct'] = abs(drift_info['test_mean'] - drift_info['train_mean']) / abs(drift_info['train_mean']) * 100
        else:
            drift_info['mean_drift_pct'] = 0
        
        # Kolmogorov-Smirnov statistic would be good here too
        drift_analysis[col] = drift_info
    
    return drift_analysis


def print_data_quality_report(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Print formatted data quality report.
    
    Args:
        df: DataFrame to analyze.
        name: Name of dataset.
    """
    quality = check_data_quality(df, name)
    
    print(f"\n{'='*50}")
    print(f"DATA QUALITY: {quality['name']}")
    print(f"{'='*50}")
    
    print(f"Shape: {quality['shape']}")
    print(f"Empty: {quality['empty']} | Duplicates: {quality['duplicates']} | Total NaN: {quality['total_nans']} | Infinites: {quality['total_infinites']}")
    
    if quality['columns_with_nans']:
        print("Columns with NaN:", end="")
        for col, count in quality['columns_with_nans'].items():
            pct = count / quality['shape'][0] * 100
            print(f" {col}({pct:.0f}%)", end="")
        print()
    
    # Show constant columns count but not full list (too verbose)
    const_count = len(quality['constant_columns'])
    if const_count > 0:
        print(f"Constant columns: {const_count}")
    
    if 'label_distribution' in quality:
        dist = quality['label_distribution']
        print(f"Label dist: {dict(sorted(dist.items()))} | Imbalance ratio: {quality['label_balance_ratio']:.2f}")
    
    print(f"Quality Score: {quality['quality_score']:.0%}")
    print(f"{'='*50}\n")
    
    return quality


def print_drift_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold_pct: float = 10.0
):
    """
    Print feature drift report highlighting significant changes.
    
    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        threshold_pct: Highlight drifts above this percentage.
    """
    drift = detect_feature_drift(train_df, test_df)
    
    print(f"\n{'='*60}")
    print(f"FEATURE DRIFT ANALYSIS (threshold: {threshold_pct}%)")
    print(f"{'='*60}\n")
    
    significant_drifts = []
    
    for col, metrics in sorted(drift.items()):
        drift_pct = metrics['mean_drift_pct']
        is_significant = drift_pct > threshold_pct
        
        if is_significant:
            print(f"[HIGH DRIFT] {col}")
            print(f"  Train mean: {metrics['train_mean']:.4f} ± {metrics['train_std']:.4f}")
            print(f"  Test mean:  {metrics['test_mean']:.4f} ± {metrics['test_std']:.4f}")
            print(f"  Drift:      {drift_pct:.2f}%")
            print()
            significant_drifts.append((col, drift_pct))
    
    if not significant_drifts:
        print(f"No significant feature drift detected (>{threshold_pct}%)")
    else:
        print(f"Found {len(significant_drifts)} features with significant drift")
    
    print(f"{'='*60}\n")
    
    return significant_drifts
