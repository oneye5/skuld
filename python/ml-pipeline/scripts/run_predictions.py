"""Run predictions on current data for real-world use.

This script generates stock rankings/predictions for investment decisions.
It trains a model on all available historical data and produces predictions
for the most recent timestamp (or a specified future timestamp).

Two modes of operation:

1. TRAIN AND PREDICT (default):
   Train a fresh model on all historical data, then generate predictions.
   Use this for periodic (e.g., monthly) rebalancing decisions.
   
   uv run python scripts/run_predictions.py

2. LOAD AND PREDICT:
   Load a previously saved model and generate predictions.
   Use this for quick predictions without retraining.
   
   uv run python scripts/run_predictions.py --load-model output/models/latest.pkl

Output:
   - Console: Top N and Bottom N ranked stocks with scores
   - CSV: Full predictions saved to output/predictions/predictions_YYYYMMDD.csv
   - Model: Optionally saved for reuse

Important Notes:
   - The model predicts RELATIVE performance (ranking), not absolute returns
   - Predictions are for the configured forward horizon (default 365 days)
   - Ensure your data is up-to-date before running predictions
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experiment_tracking import get_git_info

from config.columns import TIMESTAMP, TICKER, CLOSE
from config.settings import (
    FORWARD_RETURN_DAYS,
    RANKER_N_ESTIMATORS,
    RANKER_LEARNING_RATE,
    RANKER_NUM_LEAVES,
    RANKER_MAX_DEPTH,
    RANKER_MIN_CHILD_SAMPLES,
    RANKER_SUBSAMPLE,
    RANKER_COLSAMPLE_BYTREE,
    RANKER_DEVICE,
)
from core.logging_config import setup_logging, get_logger, log_timing


logger = get_logger(__name__)


def save_predictions_with_metadata(
    result_df: "pd.DataFrame",
    csv_path: Path,
    prediction_date: datetime,
    forward_days: int,
    training_samples: int,
    n_features: int,
    model_config: dict | None = None,
) -> None:
    """Save predictions CSV with metadata header for traceability.
    
    The CSV includes comment lines (prefixed with #) at the top containing:
    - Generation timestamp
    - Target prediction date (when predictions should be evaluated)
    - Git commit and branch information
    - Model configuration summary
    
    To read the CSV in Python, use:
        pd.read_csv(path, comment='#')
    
    Args:
        result_df: DataFrame with predictions to save.
        csv_path: Path to save the CSV file.
        prediction_date: Date the predictions are made for.
        forward_days: Forward return horizon.
        training_samples: Number of samples used in training.
        n_features: Number of features used.
        model_config: Optional model configuration dict.
    """
    import pandas as pd
    
    # Get git info for traceability
    git_info = get_git_info()
    
    # Compute target date (when predictions should be evaluated)
    target_date = prediction_date + timedelta(days=forward_days)
    
    # Build metadata header lines (prefixed with #)
    metadata_lines = [
        "# SKULD PREDICTION OUTPUT",
        "#",
        f"# Generated:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Prediction Date:    {prediction_date.strftime('%Y-%m-%d')}",
        f"# Target Date:        ~{target_date.strftime('%Y-%m-%d')} (+{forward_days} days)",
        "#",
        f"# Git Commit:         {git_info.get('git_commit', 'unknown') or 'unknown'}",
        f"# Git Branch:         {git_info.get('git_branch', 'unknown') or 'unknown'}",
        f"# Uncommitted Changes:{' Yes' if git_info.get('git_dirty') else ' No'}",
        "#",
        f"# Training Samples:   {training_samples:,}",
        f"# Features Used:      {n_features}",
        f"# Stocks Ranked:      {len(result_df)}",
    ]
    
    # Add model config if provided
    if model_config:
        metadata_lines.append("#")
        metadata_lines.append("# Model Config:")
        for key, value in model_config.items():
            if key not in ('device',):  # Skip non-essential config
                metadata_lines.append(f"#   {key}: {value}")
    
    metadata_lines.append("#")
    metadata_lines.append("# " + "=" * 50)
    metadata_lines.append("")
    
    # Write metadata header followed by CSV data
    with open(csv_path, 'w', newline='') as f:
        f.write('\n'.join(metadata_lines))
        result_df.to_csv(f, index=False)


def train_and_predict(
    top_n: int = 10,
    bottom_n: int = 10,
    forward_days: int = FORWARD_RETURN_DAYS,
    save_model_path: str | None = None,
    output_dir: str | None = None,
) -> "PredictionResult":
    """Train a model on all historical data and generate predictions.
    
    This is the primary function for production use. It:
    1. Loads all available historical data
    2. Adds features to the data
    3. Trains a ranking model on all data (except the most recent timestamp)
    4. Generates predictions for the most recent timestamp
    5. Optionally saves the model for later reuse
    
    Args:
        top_n: Number of top stocks to highlight in output.
        bottom_n: Number of bottom stocks to highlight in output.
        forward_days: Forward return horizon the model was trained for.
        save_model_path: If provided, save the trained model to this path.
        output_dir: Directory for output files. Defaults to output/predictions/.
    
    Returns:
        PredictionResult with rankings and metadata.
    """
    import pandas as pd
    import numpy as np
    import gc
    
    from core.data_loader import load_long_data
    from core.long_to_wide import long_to_wide, add_macro_prefix, clean_and_classify_tickers
    from core.preprocessor import preprocess_data, drop_sparse_columns
    from core.scaler import fit_scaler, transform_data
    from core.target_builder import compute_forward_returns, FORWARD_RETURN
    from core.model_persistence import ModelBundle, save_model, compute_data_fingerprint
    from features.ratios import add_financial_ratios
    from features.technical import add_technical_features
    from features.alpha_factors import add_alpha_factors
    from learner.ranking import (
        LightGBMRankerWrapper,
        RankerConfig,
        build_group_from_timestamps,
        prepare_ranking_data,
        filter_min_stocks_per_timestamp,
    )
    from pipeline.ranking_pipeline import get_feature_columns_for_ranking
    from config.settings import MS_PER_DAY, YEAR_2000_MS, MIN_STOCKS_PER_TIMESTAMP

    print("=" * 60)
    print("PREDICTION PIPELINE - Train and Predict")
    print("=" * 60)
    print(f"Forward horizon:    {forward_days} days")
    print(f"Show top/bottom:    {top_n}/{bottom_n} stocks")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("\n[1/6] Loading data...")
    with log_timing("data loading", logger):
        long_df = load_long_data()
        print(f"  Loaded {len(long_df):,} rows")
        
        # Filter to recent data (post-2000)
        long_df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS]
        
        # Convert to wide format
        long_df = clean_and_classify_tickers(long_df)
        long_df = add_macro_prefix(long_df)
        wide_df = long_to_wide(long_df)
        del long_df
        gc.collect()
        
        wide_df = drop_sparse_columns(wide_df, threshold=0.95)
        print(f"  Wide format: {len(wide_df):,} rows, {len(wide_df.columns)} columns")
    
    # Step 2: Add features
    print("\n[2/6] Engineering features...")
    with log_timing("feature engineering", logger):
        wide_df = add_financial_ratios(wide_df)
        wide_df = add_technical_features(wide_df)
        wide_df = add_alpha_factors(wide_df)
        
        # Force float32
        for col in wide_df.columns:
            if wide_df[col].dtype == 'float64':
                wide_df[col] = wide_df[col].astype('float32')
        
        print(f"  After features: {len(wide_df.columns)} columns")
    
    # Step 3: Split into training and prediction sets
    print("\n[3/6] Preparing train/predict split...")
    timestamps = sorted(wide_df[TIMESTAMP].unique())
    n_timestamps = len(timestamps)
    print(f"  Total timestamps: {n_timestamps}")
    
    # For training, we need forward returns, so we can only train on data
    # where future prices exist. The latest timestamp is for prediction only.
    latest_ts = timestamps[-1]
    latest_date = datetime.fromtimestamp(latest_ts / 1000)
    print(f"  Latest timestamp: {latest_date.strftime('%Y-%m-%d')}")
    
    # Training data: include all timestamps except the prediction target.
    # compute_forward_returns will naturally drop rows that don't have valid
    # future prices (via drop_na=True), so we don't need an overly conservative
    # buffer here. The key is that price_lookup_df contains future prices.
    # We only exclude the latest timestamp which is our prediction target.
    train_timestamps = timestamps[:-1]  # All except the last (prediction) timestamp
    predict_timestamps = [latest_ts]  # Just predict for the most recent
    
    if len(train_timestamps) < 10:
        raise ValueError(
            f"Insufficient training data. Only have {len(train_timestamps)} timestamps. "
            "Need at least 10 timestamps for meaningful training."
        )
    
    print(f"  Training timestamps: {len(train_timestamps)}")
    print(f"  Prediction timestamps: {len(predict_timestamps)}")
    
    # Split the data
    train_df = wide_df[wide_df[TIMESTAMP].isin(train_timestamps)].copy()
    
    # Get each ticker's most recent row for prediction
    # (Different tickers may have different latest timestamps due to data fetch timing)
    idx_latest_per_ticker = wide_df.groupby(TICKER)[TIMESTAMP].idxmax()
    predict_df = wide_df.loc[idx_latest_per_ticker].copy()
    
    # Filter to only include recent data (within 7 days of latest)
    min_acceptable_ts = latest_ts - (7 * 24 * 60 * 60 * 1000)  # 7 days in ms
    predict_df = predict_df[predict_df[TIMESTAMP] >= min_acceptable_ts]
    
    print(f"  Training rows: {len(train_df):,}")
    print(f"  Prediction rows (stocks with recent data): {len(predict_df):,}")
    
    # Step 4: Prepare training data with forward returns
    print("\n[4/6] Computing forward returns for training...")
    with log_timing("forward returns", logger):
        # Need price lookup that includes future prices for return calculation
        train_with_returns = compute_forward_returns(
            train_df,
            lookahead_days=forward_days,
            return_type="simple",
            winsorize_limits=(-0.5, 0.5),
            drop_na=True,
            price_lookup_df=wide_df,  # Use full data for price lookup
        )
        print(f"  Training samples with valid returns: {len(train_with_returns):,}")
    
    # Filter timestamps with too few stocks
    train_with_returns = filter_min_stocks_per_timestamp(
        train_with_returns, MIN_STOCKS_PER_TIMESTAMP, TIMESTAMP
    )
    
    if train_with_returns.empty:
        raise ValueError("No valid training data after filtering")
    
    # Preprocess
    train_processed = preprocess_data(train_with_returns, add_missing_flags=False)
    predict_processed = preprocess_data(predict_df, add_missing_flags=False)
    
    # Get feature columns (intersection ensures consistency)
    train_features = set(get_feature_columns_for_ranking(train_processed))
    predict_features = set(get_feature_columns_for_ranking(predict_processed))
    feature_cols = sorted(train_features & predict_features)
    
    print(f"  Feature columns: {len(feature_cols)}")
    
    if not feature_cols:
        raise ValueError("No valid feature columns after preprocessing")
    
    # Fit scaler on training data only
    print("\n[5/6] Training ranking model...")
    scaler = fit_scaler(train_processed[feature_cols])
    train_scaled = transform_data(train_processed, scaler)
    predict_scaled = transform_data(predict_processed, scaler)
    
    # Prepare ranking data
    X_train, y_train, groups_train = prepare_ranking_data(
        train_scaled,
        feature_cols=feature_cols,
        target_col=FORWARD_RETURN,
        timestamp_col=TIMESTAMP,
    )
    
    # Configure and train ranker
    ranker_config = RankerConfig(
        n_estimators=RANKER_N_ESTIMATORS,
        learning_rate=RANKER_LEARNING_RATE,
        num_leaves=RANKER_NUM_LEAVES,
        max_depth=RANKER_MAX_DEPTH,
        min_child_samples=RANKER_MIN_CHILD_SAMPLES,
        subsample=RANKER_SUBSAMPLE,
        colsample_bytree=RANKER_COLSAMPLE_BYTREE,
        device=RANKER_DEVICE,
    )
    
    ranker = LightGBMRankerWrapper(ranker_config)
    
    with log_timing("model training", logger):
        ranker.fit(X_train, y_train, groups_train)
    
    print(f"  Trained on {len(X_train):,} samples, {len(feature_cols)} features")
    
    # Step 6: Generate predictions
    print("\n[6/6] Generating predictions...")
    
    # Prepare prediction data
    X_predict = predict_scaled[feature_cols].values
    predictions = ranker.predict(pd.DataFrame(X_predict, columns=feature_cols))
    
    # Build result DataFrame
    result_df = pd.DataFrame({
        TIMESTAMP: predict_scaled[TIMESTAMP].values,
        TICKER: predict_scaled[TICKER].values,
        "predicted_score": predictions,
        "rank": pd.Series(predictions).rank(ascending=False, method="first").astype(int).values,
    })
    
    # Sort by score (best first)
    result_df = result_df.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    
    # Add Close price for reference
    if CLOSE in predict_scaled.columns:
        close_map = predict_scaled.set_index(TICKER)[CLOSE].to_dict()
        result_df["close_price"] = result_df[TICKER].map(close_map)
    
    # Create output
    prediction_date = datetime.fromtimestamp(latest_ts / 1000)
    
    result = PredictionResult(
        predictions=result_df,
        prediction_date=prediction_date,
        forward_days=forward_days,
        n_stocks=len(result_df),
        feature_columns=feature_cols,
        model_config=ranker_config.__dict__,
        training_samples=len(X_train),
    )
    
    # Print results
    print("\n" + "=" * 60)
    print(f"PREDICTIONS FOR {prediction_date.strftime('%Y-%m-%d')}")
    print(f"Forward horizon: {forward_days} days")
    print("=" * 60)
    
    print(f"\n=== TOP {top_n} STOCKS (LONG) ===")
    top_stocks = result_df.head(top_n)
    for _, row in top_stocks.iterrows():
        price_str = f"${row['close_price']:.2f}" if 'close_price' in row and pd.notna(row['close_price']) else "N/A"
        print(f"  {row['rank']:3d}. {row[TICKER]:10s}  Score: {row['predicted_score']:7.4f}  Price: {price_str}")
    
    print(f"\n=== BOTTOM {bottom_n} STOCKS (SHORT/AVOID) ===")
    bottom_stocks = result_df.tail(bottom_n).iloc[::-1]  # Reverse for worst-first
    for _, row in bottom_stocks.iterrows():
        price_str = f"${row['close_price']:.2f}" if 'close_price' in row and pd.notna(row['close_price']) else "N/A"
        print(f"  {row['rank']:3d}. {row[TICKER]:10s}  Score: {row['predicted_score']:7.4f}  Price: {price_str}")
    
    # Save outputs
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output" / "predictions"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions CSV with metadata
    csv_filename = f"predictions_{prediction_date.strftime('%Y%m%d')}.csv"
    csv_path = output_dir / csv_filename
    save_predictions_with_metadata(
        result_df=result_df,
        csv_path=csv_path,
        prediction_date=prediction_date,
        forward_days=forward_days,
        training_samples=len(X_train),
        n_features=len(feature_cols),
        model_config=ranker_config.__dict__,
    )
    print(f"\n✓ Predictions saved to: {csv_path}")
    
    # Save model if requested
    if save_model_path:
        model_path = Path(save_model_path)
        
        bundle = ModelBundle(
            ranker=ranker,
            scaler=scaler,
            feature_columns=feature_cols,
            config={
                "forward_return_days": forward_days,
                "n_estimators": ranker_config.n_estimators,
                "learning_rate": ranker_config.learning_rate,
                "training_samples": len(X_train),
            },
            metadata={
                "prediction_date": prediction_date.isoformat(),
                "data_fingerprint": compute_data_fingerprint(train_df),
            },
        )
        
        save_model(bundle, model_path)
        print(f"✓ Model saved to: {model_path}")
    
    return result


def load_and_predict(
    model_path: str,
    top_n: int = 10,
    bottom_n: int = 10,
    output_dir: str | None = None,
) -> "PredictionResult":
    """Load a saved model and generate predictions on current data.
    
    This is faster than train_and_predict since it skips training.
    Use when you have a recently trained model and just want fresh predictions.
    
    Args:
        model_path: Path to saved model (.pkl file).
        top_n: Number of top stocks to highlight.
        bottom_n: Number of bottom stocks to highlight.
        output_dir: Directory for output files.
    
    Returns:
        PredictionResult with rankings.
    """
    import pandas as pd
    import numpy as np
    import gc
    
    from core.data_loader import load_long_data
    from core.long_to_wide import long_to_wide, add_macro_prefix, clean_and_classify_tickers
    from core.preprocessor import (
        preprocess_data, drop_sparse_columns, 
        detect_price_anomalies, filter_anomalous_data
    )
    from core.scaler import transform_data
    from core.model_persistence import load_model
    from features.ratios import add_financial_ratios
    from features.technical import add_technical_features
    from features.alpha_factors import add_alpha_factors
    from pipeline.ranking_pipeline import get_feature_columns_for_ranking
    from config.settings import YEAR_2000_MS, FILTER_ANOMALIES, ANOMALY_RETURN_THRESHOLD
    from config.columns import CLOSE, TICKER
    
    print("=" * 60)
    print("PREDICTION PIPELINE - Load and Predict")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print("=" * 60)
    
    # Load model
    print("\n[1/4] Loading saved model...")
    bundle = load_model(model_path)
    print(bundle.summary())
    
    forward_days = bundle.config.get("forward_return_days", FORWARD_RETURN_DAYS)
    
    # Load and prepare data
    print("\n[2/4] Loading current data...")
    with log_timing("data loading", logger):
        long_df = load_long_data()
        long_df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS]
        long_df = clean_and_classify_tickers(long_df)
        long_df = add_macro_prefix(long_df)
        wide_df = long_to_wide(long_df)
        del long_df
        gc.collect()
        wide_df = drop_sparse_columns(wide_df, threshold=0.95)
        
        # Filter price anomalies (consistent with training pipeline)
        if FILTER_ANOMALIES and CLOSE in wide_df.columns:
            wide_df = detect_price_anomalies(
                wide_df,
                price_col=CLOSE,
                return_threshold=ANOMALY_RETURN_THRESHOLD,
            )
            wide_df, _ = filter_anomalous_data(wide_df, trim_before_anomaly=True)
    
    # Add features
    print("\n[3/4] Engineering features...")
    with log_timing("feature engineering", logger):
        wide_df = add_financial_ratios(wide_df)
        wide_df = add_technical_features(wide_df)
        wide_df = add_alpha_factors(wide_df)
        
        for col in wide_df.columns:
            if wide_df[col].dtype == 'float64':
                wide_df[col] = wide_df[col].astype('float32')
    
    # Get each ticker's most recent row for prediction
    # (Different tickers may have different latest timestamps due to data fetch timing)
    latest_ts = wide_df[TIMESTAMP].max()
    latest_date = datetime.fromtimestamp(latest_ts / 1000)
    
    # Get the most recent row per ticker
    idx_latest_per_ticker = wide_df.groupby(TICKER)[TIMESTAMP].idxmax()
    predict_df = wide_df.loc[idx_latest_per_ticker].copy()
    
    # Filter to only include recent data (within 7 days of latest)
    min_acceptable_ts = latest_ts - (7 * 24 * 60 * 60 * 1000)  # 7 days in ms
    predict_df = predict_df[predict_df[TIMESTAMP] >= min_acceptable_ts]
    
    print(f"  Latest data date: {latest_date.strftime('%Y-%m-%d')}")
    print(f"  Stocks with recent data: {len(predict_df)}")
    
    # Preprocess and scale
    print("\n[4/4] Generating predictions...")
    predict_processed = preprocess_data(predict_df, add_missing_flags=False)
    
    # Check that all required features are present
    missing_features = set(bundle.feature_columns) - set(predict_processed.columns)
    if missing_features:
        # Critical: missing features cause misaligned predictions
        # The model expects specific features in a specific order
        raise ValueError(
            f"Cannot generate predictions: {len(missing_features)} required features are missing.\n"
            f"Missing features: {sorted(missing_features)[:20]}{'...' if len(missing_features) > 20 else ''}\n"
            "This typically happens when:\n"
            "  1. The data has changed since the model was trained\n"
            "  2. Feature engineering code has been modified\n"
            "  3. The model was trained on a different data source\n"
            "Consider retraining the model with --save-model or use train_and_predict instead."
        )
    
    # Scale using saved scaler (validates all required columns exist)
    predict_scaled = transform_data(predict_processed, bundle.scaler, strict=True)
    
    # Validate prediction data is not empty
    if len(predict_scaled) == 0:
        raise ValueError(
            "No valid data for prediction after preprocessing. "
            "Check that your data file contains recent data."
        )
    
    # Generate predictions using exact feature columns from training
    X_predict = predict_scaled[bundle.feature_columns].values
    
    if np.isnan(X_predict).all():
        raise ValueError(
            "All prediction features are NaN after preprocessing. "
            "This may indicate data quality issues or incompatible preprocessing."
        )
    
    predictions = bundle.ranker.predict(pd.DataFrame(X_predict, columns=bundle.feature_columns))
    
    # Build result DataFrame
    result_df = pd.DataFrame({
        TIMESTAMP: predict_scaled[TIMESTAMP].values,
        TICKER: predict_scaled[TICKER].values,
        "predicted_score": predictions,
        "rank": pd.Series(predictions).rank(ascending=False, method="first").astype(int).values,
    })
    
    result_df = result_df.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    
    # Add Close price
    if CLOSE in predict_scaled.columns:
        close_map = predict_scaled.set_index(TICKER)[CLOSE].to_dict()
        result_df["close_price"] = result_df[TICKER].map(close_map)
    
    result = PredictionResult(
        predictions=result_df,
        prediction_date=latest_date,
        forward_days=forward_days,
        n_stocks=len(result_df),
        feature_columns=bundle.feature_columns,
        model_config=bundle.config,
        training_samples=bundle.config.get("training_samples", 0),
    )
    
    # Print results
    print("\n" + "=" * 60)
    print(f"PREDICTIONS FOR {latest_date.strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    print(f"\n=== TOP {top_n} STOCKS (LONG) ===")
    top_stocks = result_df.head(top_n)
    for _, row in top_stocks.iterrows():
        price_str = f"${row['close_price']:.2f}" if 'close_price' in row and pd.notna(row['close_price']) else "N/A"
        print(f"  {row['rank']:3d}. {row[TICKER]:10s}  Score: {row['predicted_score']:7.4f}  Price: {price_str}")
    
    print(f"\n=== BOTTOM {bottom_n} STOCKS (SHORT/AVOID) ===")
    bottom_stocks = result_df.tail(bottom_n).iloc[::-1]
    for _, row in bottom_stocks.iterrows():
        price_str = f"${row['close_price']:.2f}" if 'close_price' in row and pd.notna(row['close_price']) else "N/A"
        print(f"  {row['rank']:3d}. {row[TICKER]:10s}  Score: {row['predicted_score']:7.4f}  Price: {price_str}")
    
    # Save outputs
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output" / "predictions"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_filename = f"predictions_{latest_date.strftime('%Y%m%d')}.csv"
    csv_path = output_dir / csv_filename
    save_predictions_with_metadata(
        result_df=result_df,
        csv_path=csv_path,
        prediction_date=latest_date,
        forward_days=forward_days,
        training_samples=bundle.config.get("training_samples", 0),
        n_features=len(bundle.feature_columns),
        model_config=bundle.config,
    )
    print(f"\n✓ Predictions saved to: {csv_path}")
    
    return result


class PredictionResult:
    """Container for prediction pipeline output."""
    
    def __init__(
        self,
        predictions: "pd.DataFrame",
        prediction_date: datetime,
        forward_days: int,
        n_stocks: int,
        feature_columns: list,
        model_config: dict,
        training_samples: int,
    ):
        self.predictions = predictions
        self.prediction_date = prediction_date
        self.forward_days = forward_days
        self.n_stocks = n_stocks
        self.feature_columns = feature_columns
        self.model_config = model_config
        self.training_samples = training_samples
    
    @property
    def top_picks(self) -> "pd.DataFrame":
        """Get top 10 ranked stocks."""
        return self.predictions.head(10)
    
    @property
    def bottom_picks(self) -> "pd.DataFrame":
        """Get bottom 10 ranked stocks."""
        return self.predictions.tail(10)
    
    def get_stock_rank(self, ticker: str) -> dict | None:
        """Get ranking info for a specific stock."""
        match = self.predictions[self.predictions[TICKER] == ticker]
        if match.empty:
            return None
        row = match.iloc[0]
        return {
            "ticker": ticker,
            "rank": int(row["rank"]),
            "score": float(row["predicted_score"]),
            "percentile": 100 * (1 - row["rank"] / self.n_stocks),
        }


def main():
    """Main entry point for prediction pipeline."""
    setup_logging(level=logging.INFO, console=True)
    
    parser = argparse.ArgumentParser(
        description="Generate stock predictions using ranking model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train fresh model and predict
  uv run python scripts/run_predictions.py
  
  # Train and save model for reuse
  uv run python scripts/run_predictions.py --save-model output/models/latest.pkl
  
  # Load saved model and predict
  uv run python scripts/run_predictions.py --load-model output/models/latest.pkl
  
  # Show more stocks in output
  uv run python scripts/run_predictions.py --top-n 20 --bottom-n 20
""",
    )
    
    parser.add_argument(
        "--load-model", type=str, default=None,
        help="Path to saved model to load (skips training)"
    )
    parser.add_argument(
        "--save-model", type=str, default=None,
        help="Path to save trained model for later reuse"
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Number of top stocks to display (default: 10)"
    )
    parser.add_argument(
        "--bottom-n", type=int, default=10,
        help="Number of bottom stocks to display (default: 10)"
    )
    parser.add_argument(
        "--forward-days", type=int, default=FORWARD_RETURN_DAYS,
        help=f"Forward return horizon in days (default: {FORWARD_RETURN_DAYS})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output files (default: output/predictions/)"
    )
    
    args = parser.parse_args()
    
    if args.load_model:
        # Load existing model and predict
        result = load_and_predict(
            model_path=args.load_model,
            top_n=args.top_n,
            bottom_n=args.bottom_n,
            output_dir=args.output_dir,
        )
    else:
        # Train fresh model and predict
        result = train_and_predict(
            top_n=args.top_n,
            bottom_n=args.bottom_n,
            forward_days=args.forward_days,
            save_model_path=args.save_model,
            output_dir=args.output_dir,
        )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Prediction date:  {result.prediction_date.strftime('%Y-%m-%d')}")
    print(f"Forward horizon:  {result.forward_days} days")
    print(f"Stocks ranked:    {result.n_stocks}")
    print(f"Features used:    {len(result.feature_columns)}")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    main()
