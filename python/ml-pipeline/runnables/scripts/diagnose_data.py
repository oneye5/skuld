"""Diagnostic script to analyze data before model training."""

import sys
from pathlib import Path

# Add paths for hyphenated directories
_ml_pipeline = Path(__file__).parent.parent
sys.path.insert(0, str(_ml_pipeline))
sys.path.insert(0, str(_ml_pipeline / "data-preparation"))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "transformations"))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "long-to-wide"))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "data-splitting" / "train-test"))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "labeling"))

import pandas as pd
import numpy as np

from config.column_names import TIMESTAMP, TICKER, TARGET, CLOSE
from config.model_config import LOOKAHEAD_DAYS, GAIN_THRESHOLD_PCT, MS_PER_DAY
from utils.data_loader import load_long_data
from macro_prefix import add_macro_prefix
from converter import long_to_wide
from splitter import split_by_timestamp
from labeler import create_labels
from imputation import compute_imputation_stats, impute_data
from feature_engineering import add_cyclical_time_features
from scaling import fit_scalers, transform_data
from learner.trainer import get_feature_columns


def analyze_target_distribution(df: pd.DataFrame) -> dict:
    """Analyze target variable distribution."""
    if TARGET not in df.columns:
        return {"error": "No target column"}
    
    target = df[TARGET]
    return {
        "total_samples": len(target),
        "positive_class": int(target.sum()),
        "negative_class": int((target == 0).sum()),
        "positive_ratio": float(target.mean()),
        "class_imbalance_ratio": float((target == 0).sum() / max(target.sum(), 1)),
    }


def analyze_feature_distributions(df: pd.DataFrame, sample_features: int = 10) -> dict:
    """Analyze feature distributions."""
    feature_cols = get_feature_columns(df)
    
    stats = {}
    for col in feature_cols[:sample_features]:
        data = df[col].dropna()
        if len(data) == 0:
            continue
        stats[col] = {
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
            "median": float(data.median()),
            "skew": float(data.skew()) if len(data) > 2 else 0,
            "missing_pct": float(df[col].isna().mean() * 100),
        }
    return stats


def analyze_feature_target_correlations(df: pd.DataFrame, top_n: int = 20) -> dict:
    """Analyze correlations between features and target."""
    if TARGET not in df.columns:
        return {"error": "No target column"}
    
    feature_cols = get_feature_columns(df)
    correlations = {}
    
    for col in feature_cols:
        if df[col].isna().all():
            continue
        corr = df[col].corr(df[TARGET])
        if not np.isnan(corr):
            correlations[col] = float(corr)
    
    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    return {
        "top_positive": [(k, v) for k, v in sorted_corrs if v > 0][:top_n],
        "top_negative": [(k, v) for k, v in sorted_corrs if v < 0][:top_n],
        "near_zero": [(k, v) for k, v in sorted_corrs if abs(v) < 0.01][:top_n],
    }


def analyze_price_patterns(df: pd.DataFrame) -> dict:
    """Analyze price patterns that might cause issues."""
    if CLOSE not in df.columns:
        return {"error": "No Close column"}
    
    # Check for price clustering
    close_prices = df[CLOSE].dropna()
    
    # Look at price change distribution
    price_changes = close_prices.groupby(df[TICKER]).pct_change()
    
    return {
        "avg_close_price": float(close_prices.mean()),
        "median_close_price": float(close_prices.median()),
        "price_std": float(close_prices.std()),
        "avg_daily_return": float(price_changes.mean() * 100),
        "return_std": float(price_changes.std() * 100),
        "positive_days_pct": float((price_changes > 0).mean() * 100),
    }


def analyze_lookahead_bias(wide_df: pd.DataFrame, train_end_ts: int) -> dict:
    """Check for potential lookahead bias in data preparation."""
    # Get data around train/test boundary
    boundary_window = 30 * MS_PER_DAY  # 30 days
    
    train_end = wide_df[wide_df[TIMESTAMP] < train_end_ts]
    test_start = wide_df[
        (wide_df[TIMESTAMP] >= train_end_ts) & 
        (wide_df[TIMESTAMP] < train_end_ts + boundary_window)
    ]
    
    # Check for feature leakage - features that suddenly change at boundary
    feature_cols = get_feature_columns(wide_df)
    suspicious_features = []
    
    for col in feature_cols[:50]:  # Check first 50 features
        if col not in train_end.columns or train_end[col].isna().all():
            continue
        train_mean = train_end[col].mean()
        test_mean = test_start[col].mean() if len(test_start) > 0 else train_mean
        
        if train_mean != 0:
            pct_change = abs(test_mean - train_mean) / abs(train_mean) * 100
            if pct_change > 50:  # More than 50% change
                suspicious_features.append((col, pct_change))
    
    return {
        "train_samples_at_boundary": len(train_end),
        "test_samples_at_boundary": len(test_start),
        "suspicious_features": suspicious_features[:10],
    }


def run_diagnosis():
    """Run full diagnostic analysis."""
    print("=" * 60)
    print("DATA DIAGNOSIS FOR ML PIPELINE")
    print("=" * 60)
    
    # Load and prepare data
    print("\n1. Loading data...")
    long_df = load_long_data()
    print(f"   Loaded {len(long_df):,} rows")
    
    # Basic data stats
    print(f"   Unique tickers: {long_df['ticker'].nunique()}")
    print(f"   Unique features: {long_df['feature'].nunique()}")
    print(f"   Macro rows: {(long_df['ticker'] == '').sum():,}")
    
    # Prepare wide format
    print("\n2. Converting to wide format...")
    df = add_macro_prefix(long_df)
    wide_df = long_to_wide(df)
    print(f"   Wide format: {len(wide_df):,} rows, {len(wide_df.columns)} columns")
    
    # Sample feature columns
    feature_cols = [c for c in wide_df.columns if c not in [TIMESTAMP, TICKER]]
    print(f"   Sample features: {feature_cols[:5]}")
    
    # Split data
    print("\n3. Creating train/test split...")
    max_ts = wide_df[TIMESTAMP].max()
    min_ts = wide_df[TIMESTAMP].min()
    # Use middle 80% as train, last 20% as test
    train_end_ts = min_ts + int((max_ts - min_ts) * 0.8)
    test_end_ts = max_ts
    
    split = split_by_timestamp(wide_df, train_end_ts, test_end_ts=test_end_ts)
    print(f"   Train samples: {len(split.train):,}")
    print(f"   Test samples: {len(split.test):,}")
    
    # Create labels
    print("\n4. Creating labels...")
    train_labeled = create_labels(split.train, LOOKAHEAD_DAYS, GAIN_THRESHOLD_PCT)
    test_labeled = create_labels(split.test, LOOKAHEAD_DAYS, GAIN_THRESHOLD_PCT, 
                                  price_lookup_df=wide_df)
    print(f"   Labeled train: {len(train_labeled):,}")
    print(f"   Labeled test: {len(test_labeled):,}")
    
    # Target distribution
    print("\n5. Target distribution analysis...")
    target_stats = analyze_target_distribution(train_labeled)
    for k, v in target_stats.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")
    
    # Impute and scale
    print("\n6. Imputing and scaling...")
    imputation_stats = compute_imputation_stats(train_labeled)
    train_imputed = impute_data(train_labeled, imputation_stats, add_indicators=False)
    train_features = add_cyclical_time_features(train_imputed)
    
    scaler_set = fit_scalers(train_features)
    train_scaled = transform_data(train_features, scaler_set)
    
    # Feature distributions after scaling
    print("\n7. Feature distributions (after scaling)...")
    feature_stats = analyze_feature_distributions(train_scaled)
    for feat, stats in list(feature_stats.items())[:5]:
        print(f"   {feat}:")
        print(f"      mean={stats['mean']:.3f}, std={stats['std']:.3f}, skew={stats['skew']:.3f}")
    
    # Feature-target correlations
    print("\n8. Feature-target correlations...")
    correlations = analyze_feature_target_correlations(train_scaled)
    
    print("   Top positive correlations:")
    for feat, corr in correlations.get("top_positive", [])[:5]:
        print(f"      {feat}: {corr:.4f}")
    
    print("   Top negative correlations:")
    for feat, corr in correlations.get("top_negative", [])[:5]:
        print(f"      {feat}: {corr:.4f}")
    
    # Price patterns
    print("\n9. Price pattern analysis...")
    price_patterns = analyze_price_patterns(train_labeled)
    for k, v in price_patterns.items():
        print(f"   {k}: {v:.4f}")
    
    # Lookahead bias check
    print("\n10. Lookahead bias check...")
    bias_check = analyze_lookahead_bias(wide_df, train_end_ts)
    print(f"   Suspicious features at boundary: {len(bias_check['suspicious_features'])}")
    for feat, change in bias_check['suspicious_features'][:5]:
        print(f"      {feat}: {change:.1f}% change")
    
    # Key observations
    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS")
    print("=" * 60)
    
    # Class imbalance
    pos_ratio = target_stats.get("positive_ratio", 0)
    if pos_ratio > 0.6 or pos_ratio < 0.4:
        print(f"⚠️  Class imbalance: {pos_ratio:.1%} positive samples")
    else:
        print(f"✓  Class balance OK: {pos_ratio:.1%} positive samples")
    
    # Weak correlations
    top_corr = correlations.get("top_positive", [(None, 0)])[0][1] if correlations.get("top_positive") else 0
    if abs(top_corr) < 0.05:
        print(f"⚠️  Weak correlations: best is {top_corr:.4f}")
    else:
        print(f"✓  Some correlation: best is {top_corr:.4f}")
    
    # Feature count
    num_features = len(get_feature_columns(train_scaled))
    if num_features > 500:
        print(f"⚠️  Many features ({num_features}): may cause overfitting")
    else:
        print(f"✓  Feature count OK: {num_features}")
    
    return {
        "target_stats": target_stats,
        "correlations": correlations,
        "feature_stats": feature_stats,
        "price_patterns": price_patterns,
    }


if __name__ == "__main__":
    results = run_diagnosis()
