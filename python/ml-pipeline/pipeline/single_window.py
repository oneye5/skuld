"""Single window pipeline - runs one train/test cycle."""

from dataclasses import dataclass
import gc

import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER, TARGET, CLOSE
from config.settings import LOOKAHEAD_DAYS, GAIN_THRESHOLD_PCT, YEAR_2000_MS

from core.long_to_wide import long_to_wide, add_macro_prefix
from core.labeler import create_labels
from core.splitter import split_by_timestamp, TrainTestSplit
from core.scaler import fit_scaler, transform_data
from core.preprocessor import preprocess_data

from features.ratios import add_financial_ratios
from features.time_features import add_time_features
from features.ticker_encoding import encode_tickers_separately

from learner.trainer import train_model, get_feature_columns
from learner.predictor import predict


@dataclass
class SingleWindowResult:
    """Result of running a single train/test window."""
    window_id: int
    split: TrainTestSplit
    predictions: pd.DataFrame
    actuals: pd.DataFrame
    feature_cols: list[str]


def prepare_wide_data(long_df: pd.DataFrame) -> pd.DataFrame:
    """Convert long format to wide format with basic preparation.
    
    Args:
        long_df: Long format DataFrame.
    
    Returns:
        Wide format DataFrame.
    """
    # Filter out data before year 2000
    df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS].copy()
    
    # Add macro prefix and convert to wide
    df = add_macro_prefix(df)
    wide_df = long_to_wide(df)
    del df
    gc.collect()
    
    # MEMORY OPTIMIZATION: Drop columns that are >95% missing
    # This is more lenient than 50%, keeping useful macro features
    from core.preprocessor import drop_sparse_columns
    initial_cols = len(wide_df.columns)
    wide_df = drop_sparse_columns(wide_df, threshold=0.95)
    dropped = initial_cols - len(wide_df.columns)
    if dropped > 0:
        print(f"  Dropped {dropped} extremely sparse columns (>95% missing)")
    
    # Force float32 for all numeric columns to save memory
    for col in wide_df.columns:
        if wide_df[col].dtype == 'float64':
            wide_df[col] = wide_df[col].astype('float32')
    
    return wide_df


def run_single_window(
    wide_df: pd.DataFrame,
    train_end_ts: int,
    test_end_ts: int,
    window_id: int = 0,
    lookahead_days: int = LOOKAHEAD_DAYS,
    gain_threshold_pct: float = GAIN_THRESHOLD_PCT,
) -> SingleWindowResult | None:
    """Run the pipeline for a single train/test window.
    
    Args:
        wide_df: Wide format data.
        train_end_ts: End timestamp for training data (exclusive).
        test_end_ts: End timestamp for test data (exclusive).
        window_id: Identifier for this window.
        lookahead_days: Days to look ahead for labeling.
        gain_threshold_pct: Threshold for positive class.
    
    Returns:
        SingleWindowResult or None if insufficient data.
    """
    from config.settings import MS_PER_DAY
    
    lookahead_ms = lookahead_days * MS_PER_DAY
    
    # Slice data to relevant range + lookahead buffer
    buffer_end = test_end_ts + lookahead_ms
    wide_df_slice = wide_df[wide_df[TIMESTAMP] < buffer_end]
    
    # Split data (no copy needed, split_by_timestamp handles it)
    split = split_by_timestamp(wide_df_slice, train_end_ts, test_end_ts)
    
    if split.train.empty or split.test.empty:
        return None
    
    # Save original ticker/timestamp info for actuals before any transformations
    test_meta = split.test[[TIMESTAMP, TICKER, CLOSE]].copy()
    
    # Create labels (before other transformations)
    train_labeled = create_labels(
        split.train, lookahead_days, gain_threshold_pct
    )
    
    # For test, use full slice for price lookup (includes future data)
    test_labeled = create_labels(
        split.test, lookahead_days, gain_threshold_pct,
        price_lookup_df=wide_df_slice
    )
    
    # Free memory
    del wide_df_slice
    gc.collect()
    
    if train_labeled.empty or test_labeled.empty:
        return None
    
    # Store test actuals before transformations
    actuals = test_labeled[[TIMESTAMP, TICKER, TARGET, CLOSE]].copy()
    
    # Add financial ratios
    train_features = add_financial_ratios(train_labeled)
    test_features = add_financial_ratios(test_labeled)
    del train_labeled, test_labeled
    gc.collect()
    
    # Preprocess (handle NaN, infinities)
    train_processed = preprocess_data(train_features)
    test_processed = preprocess_data(test_features)
    del train_features, test_features
    gc.collect()
    
    # Ensure test has same columns as train (train is reference since test may have missing cols)
    common_cols = [c for c in train_processed.columns if c in test_processed.columns]
    train_processed = train_processed[common_cols].copy()
    test_processed = test_processed[common_cols].copy()
    gc.collect()
    
    # Add time features
    time_min = min(train_processed[TIMESTAMP].min(), test_processed[TIMESTAMP].min())
    time_max = max(train_processed[TIMESTAMP].max(), test_processed[TIMESTAMP].max())
    train_with_time = add_time_features(train_processed, time_min, time_max)
    test_with_time = add_time_features(test_processed, time_min, time_max)
    del train_processed, test_processed
    gc.collect()
    
    # Scale continuous features (fit on train only for proper ML practice)
    # Note: nzx-predictor fit on combined, but fitting on train is more correct
    scaler_set = fit_scaler(train_with_time)
    
    train_scaled = transform_data(train_with_time, scaler_set)
    test_scaled = transform_data(test_with_time, scaler_set)
    del train_with_time, test_with_time
    gc.collect()
    
    # Save ticker and timestamp before one-hot encoding removes them
    test_ticker_info = test_scaled[[TIMESTAMP, TICKER]].copy()
    
    # One-hot encode tickers SEPARATELY (avoids memory explosion from concat)
    train_encoded, test_encoded = encode_tickers_separately(train_scaled, test_scaled)
    del train_scaled, test_scaled
    gc.collect()
    
    # Get feature columns
    feature_cols = get_feature_columns(train_encoded)
    
    # Train model
    model = train_model(train_encoded, feature_cols)
    
    # Predict (pass ticker info for output)
    predictions = predict(model, test_encoded, feature_cols, ticker_info=test_ticker_info)
    del train_encoded, test_encoded
    gc.collect()
    
    return SingleWindowResult(
        window_id=window_id,
        split=split,
        predictions=predictions,
        actuals=actuals,
        feature_cols=feature_cols,
    )
