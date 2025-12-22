"""Pipeline orchestration module for running the complete ML pipeline."""

from dataclasses import dataclass

import pandas as pd

# Centralized path setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from path_setup import ML_PIPELINE_ROOT

from config.column_names import TIMESTAMP, TICKER, TARGET, CLOSE
from config.model_config import (
    LOOKAHEAD_DAYS,
    GAIN_THRESHOLD_PCT,
    MS_PER_DAY,
    USE_ENSEMBLE,
    USE_ADVANCED_FEATURES,
    USE_CROSS_SECTIONAL,
)
from config.file_paths import MODELS_DIR, SCALERS_DIR, ensure_output_dirs

from utils.data_loader import load_long_data
from macro_prefix import add_macro_prefix
from imputation import compute_imputation_stats, impute_data
from feature_engineering import add_cyclical_time_features
from technical_features import add_technical_features
from advanced_features import add_advanced_features
from cross_sectional_features import add_cross_sectional_features
from price_transforms import convert_prices_to_returns
from scaling import fit_scalers, transform_data, save_scalers
from feature_selection import select_features
from ticker_encoding import one_hot_encode_tickers
from converter import long_to_wide
from splitter import split_by_timestamp, TrainTestSplit
from labeler import create_labels
from learner import train_model, predict, save_model


@dataclass
class PipelineResult:
    """Result of running the pipeline for a single window."""
    window_id: int
    train_split: TrainTestSplit
    predictions: pd.DataFrame
    test_data_with_labels: pd.DataFrame
    feature_cols: list[str]

def prepare_wide_data(long_df: pd.DataFrame, keep_macro: bool = True) -> pd.DataFrame:
    """
    Prepare wide format data from long format.
    
    Args:
        long_df: Long format data.
        keep_macro: If True, include macro features. If False, only OHLCV for speed.
    
    Returns:
        Wide format DataFrame ready for feature engineering.
    """
    # Add macro prefix
    df = add_macro_prefix(long_df)
    
    # CRITICAL: Filter features BEFORE wide conversion to avoid memory explosion
    # The raw data has 1700+ features, most of which are sparse financial data
    # Keep only: OHLCV + optionally macro data
    ohlcv_features = {'Open', 'High', 'Low', 'Close', 'Volume'}
    
    if keep_macro:
        # Keep OHLCV and macro data
        mask = df['feature'].isin(ohlcv_features) | df['feature'].str.startswith('MACRO_')
    else:
        # Keep only OHLCV for faster processing
        mask = df['feature'].isin(ohlcv_features)
    
    df = df[mask]
    
    # Convert to wide format
    wide_df = long_to_wide(df)
    
    # Add technical features (before train/test split so all tickers have history)
    # These create momentum and trend features from price data
    wide_df = add_technical_features(wide_df)
    
    # Add advanced features if enabled (ATR, ADX, Stochastic, etc.)
    if USE_ADVANCED_FEATURES:
        wide_df = add_advanced_features(wide_df)
    
    # Add cross-sectional features if enabled (market-relative rankings)
    if USE_CROSS_SECTIONAL:
        wide_df = add_cross_sectional_features(wide_df)
    
    # Defragment DataFrame after adding many columns
    wide_df = wide_df.copy()
    
    return wide_df


def run_single_window(
    wide_df: pd.DataFrame,
    train_end_ts: int,
    test_end_ts: int,
    window_id: int,
    lookahead_days: int = LOOKAHEAD_DAYS,
    gain_threshold_pct: float = GAIN_THRESHOLD_PCT,
) -> PipelineResult | None:
    """
    Run the pipeline for a single train/test window.
    
    Args:
        wide_df: Wide format data.
        train_end_ts: End timestamp for training data (exclusive).
        test_end_ts: End timestamp for test data (exclusive).
        window_id: Identifier for this window.
        lookahead_days: Days to look ahead for labeling.
        gain_threshold_pct: Threshold for positive class.
    
    Returns:
        PipelineResult or None if insufficient data.
    """
    ensure_output_dirs()
    
    # CRITICAL OPTIMIZATION: Slice data to only relevant range + lookahead buffer
    # This reduces memory usage dramatically by not keeping the full dataset
    lookahead_ms = lookahead_days * MS_PER_DAY
    # Start from beginning of data (train uses all data before train_end_ts)
    buffer_start = wide_df[TIMESTAMP].min()
    buffer_end = test_end_ts + lookahead_ms  # Include lookahead for labeling
    
    wide_df_slice = wide_df[
        (wide_df[TIMESTAMP] >= buffer_start) & 
        (wide_df[TIMESTAMP] < buffer_end)
    ].copy()
    
    # Split data
    split = split_by_timestamp(wide_df_slice, train_end_ts, test_end_ts=test_end_ts)
    
    if split.train.empty or split.test.empty:
        return None
    
    # Create labels (before imputation to avoid leakage)
    # For training, use train data itself for price lookup
    train_labeled = create_labels(split.train, lookahead_days, gain_threshold_pct)
    
    # For test, use wide_df_slice for price lookup to access future prices beyond test_end
    test_labeled = create_labels(
        split.test, lookahead_days, gain_threshold_pct, price_lookup_df=wide_df_slice
    )
    
    if train_labeled.empty or test_labeled.empty:
        return None
    
    # CRITICAL: Convert raw prices to returns AFTER labeling
    # Labels need raw Close prices, but features should use returns
    # This prevents the model from learning "high price = already gained = won't gain more"
    train_labeled = convert_prices_to_returns(train_labeled)
    test_labeled = convert_prices_to_returns(test_labeled)
    
    # Compute imputation stats from training data
    imputation_stats = compute_imputation_stats(train_labeled)
    
    # Impute both train and test using training stats
    # add_indicators=False to reduce feature count and memory usage
    train_imputed = impute_data(train_labeled, imputation_stats, add_indicators=False)
    test_imputed = impute_data(test_labeled, imputation_stats, add_indicators=False)

    # Add cyclical time features
    train_features = add_cyclical_time_features(train_imputed)
    test_features = add_cyclical_time_features(test_imputed)
    
    # One-hot encode tickers (allows model to learn ticker-specific patterns)
    train_features = one_hot_encode_tickers(train_features)
    test_features = one_hot_encode_tickers(test_features)
    
    # Fit scalers on training data
    scaler_set = fit_scalers(train_features)
    
    # Transform both train and test
    train_scaled = transform_data(train_features, scaler_set)
    test_scaled = transform_data(test_features, scaler_set)
    
    # Feature selection - drop uninformative features
    train_selected, test_selected, dropped_features = select_features(
        train_scaled, test_scaled
    )
    
    # Save scalers
    save_scalers(scaler_set, SCALERS_DIR, window_id)
    
    # Train model
    model, feature_cols = train_model(train_selected)
    
    # Save model
    save_model(model, MODELS_DIR / f"model_window{window_id}.pkl")
    
    # Make predictions
    predictions = predict(model, test_selected, feature_cols)
    
    return PipelineResult(
        window_id=window_id,
        train_split=split,
        predictions=predictions,
        test_data_with_labels=test_labeled,
        feature_cols=feature_cols,
    )
