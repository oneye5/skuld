"""Pipeline orchestration module for running the complete ML pipeline."""

import sys
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

# Add paths for hyphenated directories
_ml_pipeline = Path(__file__).parent.parent
sys.path.insert(0, str(_ml_pipeline))
sys.path.insert(0, str(_ml_pipeline / "data-preparation"))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "transformations"))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "long-to-wide"))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "data-splitting" / "train-test"))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "labeling"))

from config.column_names import TIMESTAMP, TICKER, TARGET, CLOSE
from config.model_config import (
    LOOKAHEAD_DAYS,
    GAIN_THRESHOLD_PCT,
    MS_PER_DAY,
)
from config.file_paths import MODELS_DIR, SCALERS_DIR, ensure_output_dirs

from utils.data_loader import load_long_data
from macro_prefix import add_macro_prefix
from imputation import compute_imputation_stats, impute_data
from feature_engineering import add_cyclical_time_features
from scaling import fit_scalers, transform_data, save_scalers
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


def prepare_wide_data(long_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare wide format data from long format."""
    # Add macro prefix
    df = add_macro_prefix(long_df)
    
    # Convert to wide format
    wide_df = long_to_wide(df)
    
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
    
    # Split data
    split = split_by_timestamp(wide_df, train_end_ts, test_end_ts=test_end_ts)
    
    if split.train.empty or split.test.empty:
        return None
    
    # Create labels (before imputation to avoid leakage)
    train_labeled = create_labels(split.train, lookahead_days, gain_threshold_pct)
    test_labeled = create_labels(split.test, lookahead_days, gain_threshold_pct)
    
    if train_labeled.empty or test_labeled.empty:
        return None
    
    # Compute imputation stats from training data
    imputation_stats = compute_imputation_stats(train_labeled)
    
    # Impute both train and test using training stats
    # add_indicators=True for both to ensure feature alignment
    train_imputed = impute_data(train_labeled, imputation_stats, add_indicators=True)
    test_imputed = impute_data(test_labeled, imputation_stats, add_indicators=True)
    
    # Add cyclical time features
    train_features = add_cyclical_time_features(train_imputed)
    test_features = add_cyclical_time_features(test_imputed)
    
    # Fit scalers on training data
    scaler_set = fit_scalers(train_features)
    
    # Transform both train and test
    train_scaled = transform_data(train_features, scaler_set)
    test_scaled = transform_data(test_features, scaler_set)
    
    # Save scalers
    save_scalers(scaler_set, SCALERS_DIR, window_id)
    
    # Train model
    model, feature_cols = train_model(train_scaled)
    
    # Save model
    save_model(model, MODELS_DIR / f"model_window{window_id}.pkl")
    
    # Make predictions
    predictions = predict(model, test_scaled, feature_cols)
    
    return PipelineResult(
        window_id=window_id,
        train_split=split,
        predictions=predictions,
        test_data_with_labels=test_labeled,
        feature_cols=feature_cols,
    )
