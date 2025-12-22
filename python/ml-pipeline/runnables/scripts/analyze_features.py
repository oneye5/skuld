"""Detailed feature analysis for stock prediction model."""

import sys
from pathlib import Path

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
from scaling import fit_scalers, transform_data, get_macro_columns, get_ticker_columns
from learner.trainer import get_feature_columns


def analyze_features_by_category():
    """Analyze features grouped by category."""
    print("=" * 70)
    print("DETAILED FEATURE ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading and preparing data...")
    long_df = load_long_data()
    
    # Check what feature types exist
    print(f"\n2. Raw data feature analysis:")
    feature_counts = long_df.groupby('feature').size().sort_values(ascending=False)
    print(f"   Total unique features: {len(feature_counts)}")
    print(f"\n   Top 20 features by count:")
    for feat, count in feature_counts.head(20).items():
        print(f"      {feat}: {count:,}")
    
    # Check macro vs ticker data
    macro_rows = long_df[long_df['ticker'] == '']
    ticker_rows = long_df[long_df['ticker'] != '']
    
    print(f"\n3. Macro vs Ticker data:")
    print(f"   Macro rows: {len(macro_rows):,} ({len(macro_rows)/len(long_df)*100:.1f}%)")
    print(f"   Ticker rows: {len(ticker_rows):,} ({len(ticker_rows)/len(long_df)*100:.1f}%)")
    
    macro_features = macro_rows['feature'].unique()
    print(f"   Unique macro features: {len(macro_features)}")
    print(f"   Sample macro features: {list(macro_features[:10])}")
    
    # Convert to wide and analyze
    df = add_macro_prefix(long_df)
    wide_df = long_to_wide(df)
    
    print(f"\n4. Wide format analysis:")
    print(f"   Shape: {wide_df.shape}")
    
    macro_cols = get_macro_columns(wide_df)
    ticker_cols = get_ticker_columns(wide_df)
    
    print(f"   Macro columns: {len(macro_cols)}")
    print(f"   Ticker columns: {len(ticker_cols)}")
    
    # Analyze price columns specifically
    print(f"\n5. Price data analysis:")
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in price_cols:
        if col in wide_df.columns:
            data = wide_df[col].dropna()
            print(f"   {col}:")
            print(f"      count: {len(data):,}")
            print(f"      mean: {data.mean():.2f}")
            print(f"      median: {data.median():.2f}")
            print(f"      min: {data.min():.2f}")
            print(f"      max: {data.max():.2f}")
    
    # Split and prepare for model
    print(f"\n6. Preparing train/test split...")
    max_ts = wide_df[TIMESTAMP].max()
    min_ts = wide_df[TIMESTAMP].min()
    train_end_ts = min_ts + int((max_ts - min_ts) * 0.8)
    test_end_ts = max_ts
    
    split = split_by_timestamp(wide_df, train_end_ts, test_end_ts=test_end_ts)
    train_labeled = create_labels(split.train, LOOKAHEAD_DAYS, GAIN_THRESHOLD_PCT)
    
    # Check target distribution per ticker
    print(f"\n7. Target distribution by ticker:")
    ticker_targets = train_labeled.groupby(TICKER)[TARGET].agg(['mean', 'count'])
    ticker_targets = ticker_targets.sort_values('count', ascending=False)
    print(f"   Top 10 tickers by sample count:")
    for ticker, row in ticker_targets.head(10).iterrows():
        print(f"      {ticker}: {row['count']:.0f} samples, {row['mean']*100:.1f}% positive")
    
    # Analyze after imputation and scaling
    print(f"\n8. After imputation and scaling...")
    imputation_stats = compute_imputation_stats(train_labeled)
    train_imputed = impute_data(train_labeled, imputation_stats, add_indicators=False)
    train_features = add_cyclical_time_features(train_imputed)
    
    scaler_set = fit_scalers(train_features)
    train_scaled = transform_data(train_features, scaler_set)
    
    # Check if scaling worked
    print(f"\n9. Scaled feature distributions:")
    feature_cols = get_feature_columns(train_scaled)
    
    # Sample a few ticker features
    sample_ticker_cols = [c for c in ticker_cols[:5] if c in train_scaled.columns]
    print(f"   Ticker features (sample):")
    for col in sample_ticker_cols:
        data = train_scaled[col].dropna()
        if len(data) > 0:
            print(f"      {col}: mean={data.mean():.3f}, std={data.std():.3f}")
    
    # Sample macro features
    sample_macro_cols = macro_cols[:5]
    print(f"   Macro features (sample):")
    for col in sample_macro_cols:
        if col in train_scaled.columns:
            data = train_scaled[col].dropna()
            if len(data) > 0:
                print(f"      {col}: mean={data.mean():.3f}, std={data.std():.3f}")
    
    # Check correlations with target - separate macro vs ticker
    print(f"\n10. Feature-target correlations by type:")
    
    # Ticker feature correlations
    ticker_corrs = {}
    for col in ticker_cols[:50]:
        if col in train_scaled.columns:
            corr = train_scaled[col].corr(train_scaled[TARGET])
            if not np.isnan(corr):
                ticker_corrs[col] = corr
    
    sorted_ticker = sorted(ticker_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"   Top ticker feature correlations:")
    for feat, corr in sorted_ticker[:10]:
        print(f"      {feat}: {corr:.4f}")
    
    # Macro feature correlations
    macro_corrs = {}
    for col in macro_cols[:100]:
        if col in train_scaled.columns:
            corr = train_scaled[col].corr(train_scaled[TARGET])
            if not np.isnan(corr):
                macro_corrs[col] = corr
    
    sorted_macro = sorted(macro_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"   Top macro feature correlations:")
    for feat, corr in sorted_macro[:10]:
        print(f"      {feat}: {corr:.4f}")
    
    # KEY INSIGHT: Check if features are mostly NaN
    print(f"\n11. Feature missing rate analysis:")
    missing_rates = {}
    for col in feature_cols:
        missing_rate = train_scaled[col].isna().mean()
        missing_rates[col] = missing_rate
    
    high_missing = [(k, v) for k, v in missing_rates.items() if v > 0.9]
    low_missing = [(k, v) for k, v in missing_rates.items() if v < 0.1]
    
    print(f"   Features with >90% missing: {len(high_missing)}")
    print(f"   Features with <10% missing: {len(low_missing)}")
    
    # Check relationship between price and target
    print(f"\n12. Direct price-target relationship:")
    if CLOSE in train_scaled.columns and TARGET in train_scaled.columns:
        # Group by ticker and check if high-priced stocks behave differently
        for ticker in train_scaled[TICKER].unique()[:5]:
            ticker_data = train_scaled[train_scaled[TICKER] == ticker]
            if len(ticker_data) > 100:
                corr = ticker_data[CLOSE].corr(ticker_data[TARGET])
                print(f"   {ticker}: Close-Target correlation = {corr:.4f}")


if __name__ == "__main__":
    analyze_features_by_category()
