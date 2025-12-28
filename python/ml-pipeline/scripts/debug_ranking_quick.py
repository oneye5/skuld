"""Quick validation script for ranking pipeline development.

Runs on a 2-year subset of data to verify code correctness quickly.
Takes ~30 seconds instead of 15 minutes for the full pipeline.

Usage:
    uv run python scripts/debug_ranking_quick.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER, CLOSE, TARGET
from config.settings import MS_PER_DAY

from core.data_loader import load_long_data
from core.long_to_wide import long_to_wide, add_macro_prefix, clean_and_classify_tickers
from core.target_builder import compute_forward_returns, FORWARD_RETURN
from core.preprocessor import preprocess_data
from core.scaler import fit_scaler, transform_data

from features.technical import add_technical_features
from features.cross_sectional import add_cross_sectional_features
from features.ratios import add_financial_ratios

from learner.ranking import LightGBMRankerWrapper, build_group_from_timestamps, RankerConfig
from evaluation.ranking_metrics import RankingMetrics, compute_cross_sectional_ic_series
from evaluation.portfolio_simulator import run_portfolio_backtest, PortfolioConfig


def main():
    print("=" * 60)
    print("RANKING DEBUG SCRIPT")
    print("=" * 60)
    print("Running on 2-year subset for quick validation...\n")
    
    # Load data
    print("Loading data...")
    long_df = load_long_data()
    print(f"Total rows: {len(long_df):,}")
    
    # Take only recent 2 years for speed
    max_ts = long_df[TIMESTAMP].max()
    two_years_ms = 2 * 365 * MS_PER_DAY
    long_df = long_df[long_df[TIMESTAMP] > (max_ts - two_years_ms)]
    print(f"Subset (2 years): {len(long_df):,} rows")
    
    # Convert to wide format
    print("\nConverting to wide format...")
    from config.settings import YEAR_2000_MS
    df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS].copy()
    df = clean_and_classify_tickers(df)
    df = add_macro_prefix(df)
    wide_df = long_to_wide(df)
    print(f"Wide format: {len(wide_df):,} rows, {len(wide_df.columns)} columns")
    
    # Compute forward returns
    print("\nComputing forward returns (5-day)...")
    wide_df = compute_forward_returns(
        wide_df, 
        lookahead_days=5, 
        return_type="simple",
        winsorize_limits=(-0.5, 0.5),
        drop_na=True,
    )
    print(f"After forward returns: {len(wide_df):,} rows")
    
    # Check forward return distribution
    fr = wide_df[FORWARD_RETURN]
    print(f"\nForward return stats:")
    print(f"  Mean:   {fr.mean():.4f}")
    print(f"  Std:    {fr.std():.4f}")
    print(f"  Min:    {fr.min():.4f}")
    print(f"  Max:    {fr.max():.4f}")
    
    # Add feature engineering
    print("\nAdding technical features...")
    wide_df = add_technical_features(wide_df)
    
    print("Adding financial ratios...")
    wide_df = add_financial_ratios(wide_df)
    
    print("Adding cross-sectional features...")
    wide_df = add_cross_sectional_features(wide_df)
    
    print(f"After feature engineering: {len(wide_df.columns)} columns")
    
    # Filter timestamps with enough stocks
    min_stocks = 5
    ts_counts = wide_df.groupby(TIMESTAMP).size()
    valid_ts = ts_counts[ts_counts >= min_stocks].index
    wide_df = wide_df[wide_df[TIMESTAMP].isin(valid_ts)]
    print(f"\nAfter min-stocks filter ({min_stocks}): {len(wide_df):,} rows")
    print(f"Valid timestamps: {len(valid_ts)}")
    print(f"Avg stocks per timestamp: {wide_df.groupby(TIMESTAMP).size().mean():.1f}")
    
    # Check timestamp spacing
    timestamps = sorted(wide_df[TIMESTAMP].unique())
    ts_spacing_days = np.median(np.diff(timestamps)) / MS_PER_DAY
    print(f"\nTimestamp spacing: {ts_spacing_days:.1f} days")
    
    forward_days = 5
    
    # CRITICAL: Filter to non-overlapping periods for valid backtesting
    # With 5-day forward returns and daily data, we should only use every 5th day
    # to avoid overlapping return periods (which would inflate Sharpe)
    if ts_spacing_days < forward_days:
        print(f"Filtering to non-overlapping periods (every {forward_days} days)...")
        # Take every Nth timestamp where N = forward_days / ts_spacing_days
        step = int(forward_days / ts_spacing_days)
        timestamps = timestamps[::step]
        wide_df = wide_df[wide_df[TIMESTAMP].isin(timestamps)]
        ts_spacing_days = forward_days
        print(f"After filtering: {len(wide_df):,} rows, {len(timestamps)} timestamps")
    
    # Train/test split with GAP to avoid target leakage
    # The gap must be >= forward_days because:
    # - Train forward returns use prices up to train_end + forward_days
    # - Test should start AFTER that to avoid using same price data
    split_idx = int(len(timestamps) * 0.8)
    gap_periods = int(np.ceil(forward_days / max(ts_spacing_days, 1)))  # Number of timestamps to skip
    
    train_ts = timestamps[:split_idx]
    # Skip 'gap_periods' timestamps between train and test
    test_start_idx = split_idx + gap_periods
    test_ts = timestamps[test_start_idx:]
    
    if len(test_ts) == 0:
        print("ERROR: Not enough data after applying gap. Reduce gap or use more data.")
        return
    
    train_df = wide_df[wide_df[TIMESTAMP].isin(train_ts)].copy()
    test_df = wide_df[wide_df[TIMESTAMP].isin(test_ts)].copy()
    
    print(f"\nTrain: {len(train_df):,} rows, {len(train_ts)} timestamps")
    print(f"Gap:   {gap_periods} timestamps (to avoid target leakage)")
    print(f"Test:  {len(test_df):,} rows, {len(test_ts)} timestamps")
    
    # Get feature columns (exclude metadata and target)
    excluded_cols = {TIMESTAMP, TICKER, TARGET, FORWARD_RETURN, 'is_macro'}
    feature_cols = [
        c for c in wide_df.columns 
        if c not in excluded_cols 
        and not c.startswith("MissingFlag_")
        and pd.api.types.is_numeric_dtype(wide_df[c])
    ]
    
    # Remove raw OHLCV (use derived features instead)
    raw_ohlcv = {'Open', 'High', 'Low', CLOSE, 'Volume'}
    feature_cols = [c for c in feature_cols if c not in raw_ohlcv]
    
    print(f"\nUsing {len(feature_cols)} features")
    if len(feature_cols) <= 20:
        print(f"Features: {feature_cols}")
    else:
        print(f"Sample features: {feature_cols[:10]} ... {feature_cols[-5:]}")
    
    # Preprocess: handle NaN and infinities
    print("\nPreprocessing data...")
    train_df = preprocess_data(train_df, add_missing_flags=False)
    test_df = preprocess_data(test_df, add_missing_flags=False)
    
    # Filter to features that exist after preprocessing
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    
    if len(feature_cols) == 0:
        print("ERROR: No features available after preprocessing!")
        return
    
    # Scale features (fit on train only)
    print("Scaling features (fit on train)...")
    scaler = fit_scaler(train_df[feature_cols])
    train_df = transform_data(train_df, scaler)
    test_df = transform_data(test_df, scaler)
    
    # Sort by timestamp
    train_df = train_df.sort_values(TIMESTAMP).reset_index(drop=True)
    test_df = test_df.sort_values(TIMESTAMP).reset_index(drop=True)
    
    # Build group parameters
    train_groups = build_group_from_timestamps(train_df, TIMESTAMP)
    test_groups = build_group_from_timestamps(test_df, TIMESTAMP)
    
    print(f"\nTrain groups: {len(train_groups)}, sum={sum(train_groups)}")
    print(f"Test groups:  {len(test_groups)}, sum={sum(test_groups)}")
    
    # Train ranking model
    print("\nTraining LGBMRanker...")
    config = RankerConfig(n_estimators=50, verbose=-1)
    ranker = LightGBMRankerWrapper(config)
    
    X_train = train_df[feature_cols]
    y_train = train_df[FORWARD_RETURN]
    
    ranker.fit(X_train, y_train, train_groups)
    print("Training complete!")
    
    # Predict on test
    print("\nPredicting on test set...")
    X_test = test_df[feature_cols]
    predictions = ranker.predict(X_test)
    
    # Build predictions DataFrame
    predictions_df = pd.DataFrame({
        TIMESTAMP: test_df[TIMESTAMP].values,
        TICKER: test_df[TICKER].values,
        "predicted_score": predictions,
        "actual_return": test_df[FORWARD_RETURN].values,
    })
    
    # Compute ranking metrics
    print("\nComputing ranking metrics...")
    metrics = RankingMetrics.from_predictions(
        predictions_df,
        timestamp_col=TIMESTAMP,
        predicted_col="predicted_score",
        actual_col="actual_return",
        min_stocks=5,
    )
    
    print("\n" + metrics.summary())
    
    # Run quick portfolio backtest
    print("\n\nRunning portfolio backtest (top-3 long, bottom-3 short)...")
    portfolio_config = PortfolioConfig(
        top_n=3,
        bottom_n=3,
        transaction_cost_bps=10,
    )
    backtest = run_portfolio_backtest(
        predictions_df,
        portfolio_config,
        timestamp_col=TIMESTAMP,
        score_col="predicted_score",
        return_col="actual_return",
    )
    
    print(backtest.summary())
    
    # Calculate periods_per_year for context
    from evaluation.portfolio_simulator import infer_periods_per_year
    periods_per_year = infer_periods_per_year(pd.Series(backtest.daily_returns.index))
    print(f"\nAnnualization: {periods_per_year} periods/year (inferred from timestamps)")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    # Reality checks
    if metrics.mean_ic > 0.15:
        print("⚠️  WARNING: IC > 0.15 is suspiciously high!")
        print("    Check for: data leakage, forward-looking features, train/test overlap")
    elif metrics.mean_ic > 0.05:
        print("✓ Good IC (0.05-0.15) - model has predictive signal")
    elif metrics.mean_ic > 0.02:
        print("~ Moderate IC (0.02-0.05) - some predictive signal")
    elif metrics.mean_ic > 0:
        print("~ Weak IC (0.00-0.02) - marginal signal")
    else:
        print("✗ Negative IC - model is not working on this data subset")
    
    if backtest.sharpe_ratio > 3.0:
        print(f"⚠️  WARNING: Sharpe > 3.0 ({backtest.sharpe_ratio:.2f}) is suspiciously high!")
        print("    This may indicate: wrong annualization, overlapping returns, or data leakage")
    elif backtest.sharpe_ratio > 1.0:
        print(f"✓ Good Sharpe ratio ({backtest.sharpe_ratio:.2f})")
    elif backtest.sharpe_ratio > 0:
        print(f"~ Positive but low Sharpe ({backtest.sharpe_ratio:.2f})")
    else:
        print(f"✗ Negative Sharpe ({backtest.sharpe_ratio:.2f}) - strategy loses money")
    
    if metrics.quintile_spread > 0:
        print("✓ Positive quintile spread - long-short strategy may work")
    else:
        print("✗ Negative quintile spread - model not separating winners/losers")
    
    print("\nDebug run complete!")


if __name__ == "__main__":
    main()
