"""Generate stock ranking predictions for the latest timestamp(s).

This script trains on all historical data and generates rankings for the most
recent timestamp(s). Use this for actual predictions on new/future data.

Usage:
    uv run python scripts/generate_predictions.py
    uv run python scripts/generate_predictions.py --output my_predictions.csv
    uv run python scripts/generate_predictions.py --n-timestamps 5

Note: "Future" means the most recent data available. The model trains on all
data EXCEPT the timestamps being predicted.
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER, CLOSE
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate stock rankings for the latest timestamp(s)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-timestamps", type=int, default=1,
        help="Number of recent timestamps to predict (1 = just the latest)"
    )
    parser.add_argument(
        "--forward-days", type=int, default=5,
        help="Forward return horizon used for training"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100,
        help="Number of boosting iterations"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV file path (default: predictions_YYYYMMDD_HHMMSS.csv)"
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Number of top stocks to highlight"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("STOCK RANKING PREDICTIONS")
    print("=" * 60)
    print(f"Predicting: {args.n_timestamps} timestamp(s)")
    print(f"Forward days (training): {args.forward_days}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    long_df = load_long_data()
    print(f"Total rows: {len(long_df):,}")
    
    # Convert to wide format
    print("Converting to wide format...")
    from config.settings import YEAR_2000_MS
    df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS].copy()
    df = clean_and_classify_tickers(df)
    df = add_macro_prefix(df)
    wide_df = long_to_wide(df)
    del df, long_df
    
    print(f"Wide format: {len(wide_df):,} rows")
    
    # Get unique timestamps and split
    timestamps = sorted(wide_df[TIMESTAMP].unique())
    print(f"Total timestamps: {len(timestamps)}")
    
    # We need forward_days + 1 more timestamps after training ends for valid training data
    # So we can use all timestamps except the last (forward_days + n_timestamps) for training
    train_cutoff_idx = len(timestamps) - args.forward_days - args.n_timestamps
    
    if train_cutoff_idx < 100:  # Need reasonable amount of training data
        print(f"ERROR: Not enough historical data. Need at least {args.forward_days + args.n_timestamps + 100} timestamps.")
        return
    
    train_ts = timestamps[:train_cutoff_idx]
    predict_ts = timestamps[-args.n_timestamps:]
    
    # Extract DataFrames
    train_df = wide_df[wide_df[TIMESTAMP].isin(train_ts)].copy()
    predict_df = wide_df[wide_df[TIMESTAMP].isin(predict_ts)].copy()
    
    print(f"\nTraining data: {len(train_df):,} rows ({len(train_ts)} timestamps)")
    print(f"Prediction data: {len(predict_df):,} rows ({len(predict_ts)} timestamps)")
    
    # Add features to both
    print("\nEngineering features...")
    train_df = add_technical_features(train_df)
    train_df = add_financial_ratios(train_df)
    train_df = add_cross_sectional_features(train_df)
    
    predict_df = add_technical_features(predict_df)
    predict_df = add_financial_ratios(predict_df)
    predict_df = add_cross_sectional_features(predict_df)
    
    # Compute forward returns for training data only
    print("Computing forward returns for training...")
    train_df = compute_forward_returns(
        train_df,
        lookahead_days=args.forward_days,
        return_type="simple",
        winsorize_limits=(-0.5, 0.5),
        drop_na=True,
    )
    
    print(f"Training samples after forward return: {len(train_df):,}")
    
    # Get feature columns (exclude metadata)
    excluded_cols = {TIMESTAMP, TICKER, CLOSE, FORWARD_RETURN, 'is_macro', 
                     'Open', 'High', 'Low', 'Volume'}
    feature_cols = [
        c for c in train_df.columns
        if c not in excluded_cols
        and not c.startswith('MissingFlag_')
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    
    # Ensure predict_df has same features
    feature_cols = [c for c in feature_cols if c in predict_df.columns]
    print(f"Using {len(feature_cols)} features")
    
    # Preprocess
    print("\nPreprocessing...")
    train_df = preprocess_data(train_df, add_missing_flags=False)
    predict_df = preprocess_data(predict_df, add_missing_flags=False)
    
    # Filter feature cols again after preprocessing
    feature_cols = [c for c in feature_cols if c in train_df.columns and c in predict_df.columns]
    
    # Scale (fit on train)
    print("Scaling features...")
    scaler = fit_scaler(train_df[feature_cols])
    train_df = transform_data(train_df, scaler)
    predict_df = transform_data(predict_df, scaler)
    
    # Sort and build groups for training
    train_df = train_df.sort_values(TIMESTAMP).reset_index(drop=True)
    train_groups = build_group_from_timestamps(train_df, TIMESTAMP)
    
    # Train model
    print(f"\nTraining LGBMRanker ({args.n_estimators} estimators)...")
    config = RankerConfig(
        n_estimators=args.n_estimators,
        verbose=-1,
    )
    ranker = LightGBMRankerWrapper(config)
    
    X_train = train_df[feature_cols]
    y_train = train_df[FORWARD_RETURN]
    
    ranker.fit(X_train, y_train, train_groups)
    print("Training complete!")
    
    # Generate predictions
    print(f"\nGenerating predictions for {len(predict_df)} stocks...")
    X_predict = predict_df[feature_cols]
    scores = ranker.predict(X_predict)
    
    # Build results DataFrame
    results = pd.DataFrame({
        TIMESTAMP: predict_df[TIMESTAMP].values,
        TICKER: predict_df[TICKER].values,
        'predicted_score': scores,
    })
    
    # Add rank within each timestamp
    results['rank'] = results.groupby(TIMESTAMP)['predicted_score'].rank(ascending=False).astype(int)
    
    # Add human-readable timestamp
    results['date'] = pd.to_datetime(results[TIMESTAMP], unit='ms').dt.strftime('%Y-%m-%d')
    
    # Sort by timestamp, then rank
    results = results.sort_values([TIMESTAMP, 'rank']).reset_index(drop=True)
    
    # Display results
    for ts in predict_ts:
        ts_df = results[results[TIMESTAMP] == ts]
        date_str = ts_df['date'].iloc[0]
        
        print(f"\n{'=' * 60}")
        print(f"PREDICTIONS FOR {date_str}")
        print(f"{'=' * 60}")
        
        print(f"\n🟢 TOP {args.top_n} (LONG candidates):")
        top = ts_df.nsmallest(args.top_n, 'rank')
        for _, row in top.iterrows():
            print(f"  #{row['rank']:2d}  {row[TICKER]:<10}  score: {row['predicted_score']:+.4f}")
        
        print(f"\n🔴 BOTTOM {args.top_n} (SHORT candidates):")
        bottom = ts_df.nlargest(args.top_n, 'rank')
        for _, row in bottom.sort_values('rank', ascending=False).iterrows():
            print(f"  #{row['rank']:2d}  {row[TICKER]:<10}  score: {row['predicted_score']:+.4f}")
    
    # Save to file
    if args.output:
        output_path = args.output
    else:
        output_path = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Reorder columns for output
    output_cols = ['date', TIMESTAMP, 'rank', TICKER, 'predicted_score']
    results[output_cols].to_csv(output_path, index=False)
    print(f"\n✅ Saved predictions to: {output_path}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Timestamps predicted: {len(predict_ts)}")
    print(f"Total stocks ranked: {len(results)}")
    print(f"Training samples used: {len(train_df):,}")
    print(f"Features used: {len(feature_cols)}")
    
    return results


if __name__ == "__main__":
    main()
