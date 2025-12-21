"""Diagnostic script to investigate model performance and data quality at each pipeline step."""

import json
import gc

import pandas as pd
import numpy as np

# Centralized path setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from path_setup import ML_PIPELINE_ROOT

from config.column_names import TIMESTAMP, TICKER, TARGET, CLOSE
from config.model_config import LOOKAHEAD_DAYS, GAIN_THRESHOLD_PCT, MS_PER_DAY

from utils.data_loader import load_long_data
from macro_prefix import add_macro_prefix
from converter import long_to_wide
from technical_features import add_technical_features
from splitter import split_by_timestamp
from labeler import create_labels
from price_transforms import convert_prices_to_returns
from imputation import compute_imputation_stats, impute_data
from feature_engineering import add_cyclical_time_features
from scaling import fit_scalers, transform_data


def sample_df(df: pd.DataFrame, name: str, n_samples: int = 5):
    """Print diagnostic info about a DataFrame."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns[:20])}{'...' if len(df.columns) > 20 else ''}")
    
    if TICKER in df.columns:
        print(f"Unique tickers: {df[TICKER].nunique()}")
    
    if TARGET in df.columns:
        target_vals = df[TARGET].value_counts(dropna=False)
        print(f"Target distribution:\n{target_vals}")
        print(f"Target mean: {df[TARGET].mean():.4f}")
    
    # Check for NaN
    nan_counts = df.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"Columns with NaN ({len(cols_with_nan)}): {list(cols_with_nan.head(10).index)}")
        print(f"Total NaN: {nan_counts.sum()}")
    
    # Show sample data
    print(f"\nSample data ({n_samples} rows):")
    sample_cols = [c for c in [TIMESTAMP, TICKER, CLOSE, TARGET] if c in df.columns]
    if len(sample_cols) > 0:
        print(df[sample_cols].head(n_samples).to_string())
    else:
        print(df.head(n_samples).to_string())
    
    # Numeric columns stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric columns stats (first 5):")
        for col in numeric_cols[:5]:
            if col != TIMESTAMP:
                print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, "
                      f"min={df[col].min():.4f}, max={df[col].max():.4f}")


def check_feature_correlation_with_target(df: pd.DataFrame, top_n: int = 20):
    """Check which features correlate with the target."""
    if TARGET not in df.columns:
        return
    
    print(f"\n{'='*60}")
    print("FEATURE CORRELATIONS WITH TARGET")
    print(f"{'='*60}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in [TIMESTAMP, TARGET]]
    
    correlations = {}
    for col in numeric_cols:
        try:
            corr = df[col].corr(df[TARGET])
            if not np.isnan(corr):
                correlations[col] = corr
        except:
            pass
    
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\nTop {top_n} features by absolute correlation:")
    for col, corr in sorted_corrs[:top_n]:
        print(f"  {col}: {corr:.4f}")
    
    print(f"\nBottom {top_n} features by absolute correlation:")
    for col, corr in sorted_corrs[-top_n:]:
        print(f"  {col}: {corr:.4f}")


def check_label_distribution_by_period(df: pd.DataFrame):
    """Check if target distribution varies significantly over time."""
    if TARGET not in df.columns or TIMESTAMP not in df.columns:
        return
    
    print(f"\n{'='*60}")
    print("TARGET DISTRIBUTION BY TIME PERIOD")
    print(f"{'='*60}")
    
    df = df.copy()
    df['date'] = pd.to_datetime(df[TIMESTAMP], unit='ms')
    df['year'] = df['date'].dt.year
    
    yearly_stats = df.groupby('year')[TARGET].agg(['mean', 'count'])
    print("\nTarget mean by year:")
    print(yearly_stats.to_string())


def run_diagnostics():
    """Run full diagnostic pipeline."""
    print("Loading raw data...")
    long_df = load_long_data()
    sample_df(long_df, "1. Raw Long Data")
    
    print("\nAdding macro prefix...")
    df = add_macro_prefix(long_df)
    sample_df(df, "2. After Macro Prefix")
    
    print("\nConverting to wide format...")
    wide_df = long_to_wide(df)
    sample_df(wide_df, "3. Wide Format")
    
    print("\nAdding technical features...")
    wide_df = add_technical_features(wide_df)
    sample_df(wide_df, "4. With Technical Features")
    
    # Find split timestamp (recent enough to have data)
    max_ts = wide_df[TIMESTAMP].max()
    min_ts = wide_df[TIMESTAMP].min()
    
    test_period_ms = int(1.5 * 365.25 * MS_PER_DAY)
    lookahead_ms = LOOKAHEAD_DAYS * MS_PER_DAY
    
    test_end_ts = max_ts - lookahead_ms
    train_end_ts = test_end_ts - test_period_ms
    
    print(f"\nSplit timestamps:")
    print(f"  min_ts: {pd.to_datetime(min_ts, unit='ms')}")
    print(f"  train_end: {pd.to_datetime(train_end_ts, unit='ms')}")
    print(f"  test_end: {pd.to_datetime(test_end_ts, unit='ms')}")
    print(f"  max_ts: {pd.to_datetime(max_ts, unit='ms')}")
    
    print("\nSplitting data...")
    split = split_by_timestamp(wide_df, train_end_ts, test_end_ts=test_end_ts)
    sample_df(split.train, "5a. Train Split")
    sample_df(split.test, "5b. Test Split")
    
    print("\nCreating labels...")
    train_labeled = create_labels(split.train, LOOKAHEAD_DAYS, GAIN_THRESHOLD_PCT)
    test_labeled = create_labels(
        split.test, LOOKAHEAD_DAYS, GAIN_THRESHOLD_PCT, 
        price_lookup_df=wide_df
    )
    sample_df(train_labeled, "6a. Train Labeled")
    sample_df(test_labeled, "6b. Test Labeled")
    
    check_label_distribution_by_period(train_labeled)
    
    print("\nConverting prices to returns...")
    train_returns = convert_prices_to_returns(train_labeled)
    test_returns = convert_prices_to_returns(test_labeled)
    sample_df(train_returns, "7a. Train Returns")
    sample_df(test_returns, "7b. Test Returns")
    
    print("\nImputing data...")
    imputation_stats = compute_imputation_stats(train_returns)
    train_imputed = impute_data(train_returns, imputation_stats, add_indicators=False)
    test_imputed = impute_data(test_returns, imputation_stats, add_indicators=False)
    sample_df(train_imputed, "8a. Train Imputed")
    sample_df(test_imputed, "8b. Test Imputed")
    
    print("\nAdding cyclical features...")
    train_features = add_cyclical_time_features(train_imputed)
    test_features = add_cyclical_time_features(test_imputed)
    sample_df(train_features, "9a. Train Features")
    sample_df(test_features, "9b. Test Features")
    
    print("\nScaling data...")
    scaler_set = fit_scalers(train_features)
    train_scaled = transform_data(train_features, scaler_set)
    test_scaled = transform_data(test_features, scaler_set)
    sample_df(train_scaled, "10a. Train Scaled")
    sample_df(test_scaled, "10b. Test Scaled")
    
    # Check feature correlations
    check_feature_correlation_with_target(train_scaled)
    
    # Look at feature importance by training a quick model
    print(f"\n{'='*60}")
    print("QUICK MODEL CHECK")
    print(f"{'='*60}")
    
    from config.column_names import TARGET
    from learner.trainer import get_feature_columns, train_model
    
    feature_cols = get_feature_columns(train_scaled)
    print(f"Total features: {len(feature_cols)}")
    
    model, _ = train_model(train_scaled)
    
    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 features by importance:")
    for feat, imp in sorted_imp[:20]:
        print(f"  {feat}: {imp:.4f}")
    
    # Make predictions
    from learner.predictor import predict
    preds = predict(model, test_scaled, feature_cols)
    
    print(f"\nPrediction distribution:")
    print(f"  Mean probability: {preds['prediction'].mean():.4f}")
    print(f"  Std: {preds['prediction'].std():.4f}")
    print(f"  Min: {preds['prediction'].min():.4f}")
    print(f"  Max: {preds['prediction'].max():.4f}")
    
    # Check actual vs predicted correlation
    merged = preds.merge(
        test_scaled[[TIMESTAMP, TICKER, TARGET]], 
        on=[TIMESTAMP, TICKER]
    )
    
    pred_actual_corr = merged['prediction'].corr(merged[TARGET])
    print(f"\nPrediction-Actual correlation: {pred_actual_corr:.4f}")
    
    # Check by probability buckets
    merged['prob_bucket'] = pd.cut(merged['prediction'], bins=[0, 0.3, 0.5, 0.7, 1.0])
    bucket_stats = merged.groupby('prob_bucket', observed=True)[TARGET].agg(['mean', 'count'])
    print("\nActual target rate by probability bucket:")
    print(bucket_stats.to_string())


if __name__ == "__main__":
    run_diagnostics()
