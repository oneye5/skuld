"""Post-split preprocessing with proper scaling to prevent data leakage."""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from src.config.config import *
from src.preprocessing.feature_engineering import scale_data_with_scaler
from src.utils.io_utils import load_data, save_data


def post_split_preprocessing_train(csv_in_path: str, csv_out_path: str, scaler_path: str) -> None:
    """
    Apply post-split preprocessing to training data.
    
    Fits and saves the scaler on training data so it can be applied to test data.
    Also saves column information for alignment with test data.
    This prevents data leakage and feature mismatch errors.
    
    Args:
        csv_in_path: Path to input training CSV.
        csv_out_path: Path to save scaled training CSV.
        scaler_path: Path to save the fitted RobustScaler.
    """
    df = load_data(csv_in_path)
    
    # CRITICAL: Clean NaN and infinity values BEFORE scaling
    # These can cause issues with RobustScaler and model training
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Replace infinity with NaN first (in-place where possible)
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values per ticker if available, otherwise globally
    if 'ticker' in df.columns:
        # Use groupby().transform() for vectorized per-ticker filling (much faster than explicit loop)
        for col in numeric_cols:
            df[col] = df.groupby('ticker')[col].transform(lambda x: x.ffill().bfill())
    else:
        # Ticker column not available, use global forward/backward fill
        for col in numeric_cols:
            df[col] = df[col].ffill().bfill()
    
    # As last resort, fill remaining NaNs with 0 (should be very few after ticker grouping)
    for col in numeric_cols:
        df[col] = df[col].fillna(0)
    
    # Save column info for alignment with test data
    # This ensures train and test have identical columns
    columns_path = scaler_path.replace('.pkl', '_columns.pkl')
    joblib.dump(list(df.columns), columns_path)
    
    # Fit scaler on training data
    df, scaler, continuous_cols = scale_data_with_scaler(df, scaler=None, fit_scaler=True)
    # Save continuous column info used for scaling so test uses same features
    continuous_cols_path = scaler_path.replace('.pkl', '_continuous_cols.pkl')
    joblib.dump(list(continuous_cols), continuous_cols_path)
    
    # Save scaler for later use on test data
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    
    save_data(df, csv_out_path)


def post_split_preprocessing_test(csv_in_path: str, csv_out_path: str, scaler_path: str) -> None:
    """
    Apply post-split preprocessing to test data using training scaler.
    
    Applies the scaler that was fitted on training data to prevent leakage.
    Aligns columns with training data to prevent feature mismatch errors.
    
    Missing columns in test are added with 0s. Extra columns are dropped.
    This handles cases where different tickers appear in different time periods
    and where sparse column filtering differs between time periods.
    
    Args:
        csv_in_path: Path to input test CSV.
        csv_out_path: Path to save scaled test CSV.
        scaler_path: Path to load the fitted RobustScaler.
    """
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    df = load_data(csv_in_path)
    
    # CRITICAL: Clean NaN and infinity values BEFORE scaling (same as training)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Replace infinity with NaN first (in-place where possible)
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values per ticker if available, otherwise globally
    if 'ticker' in df.columns:
        # Use groupby().transform() for vectorized per-ticker filling (much faster than explicit loop)
        for col in numeric_cols:
            df[col] = df.groupby('ticker')[col].transform(lambda x: x.ffill().bfill())
    else:
        # Ticker column not available, use global forward/backward fill
        for col in numeric_cols:
            df[col] = df[col].ffill().bfill()
    
    # As last resort, fill remaining NaNs with 0 (should be very few after ticker grouping)
    for col in numeric_cols:
        df[col] = df[col].fillna(0)
    
    # Load training column info and align test columns EXACTLY
    columns_path = scaler_path.replace('.pkl', '_columns.pkl')
    if not Path(columns_path).exists():
        raise FileNotFoundError(f"Column info not found at {columns_path}")
    
    train_columns = joblib.load(columns_path)
    test_columns = set(df.columns)
    train_columns_set = set(train_columns)
    
    # Detect mismatches
    missing_cols = train_columns_set - test_columns
    extra_cols = test_columns - train_columns_set
    
    if missing_cols or extra_cols:
        # Add missing columns with 0s - this is valid for features not present in this time period
        for col in missing_cols:
            df[col] = 0.0
        
        # Drop extra columns - these appeared in test but not train
        if extra_cols:
            df = df.drop(columns=list(extra_cols))
    
    # CRITICAL: Reorder columns to EXACTLY match train order
    # This must happen before scaling to ensure scaler gets columns in right order
    df = df[train_columns]
    
    # Load and apply fitted scaler from training data
    scaler = joblib.load(scaler_path)
    # Load continuous column info used during training so we scale the exact
    # same feature set and ordering. This prevents mismatches when binary
    # columns or sparse features appear differently across periods.
    continuous_cols_path = scaler_path.replace('.pkl', '_continuous_cols.pkl')
    if not Path(continuous_cols_path).exists():
        raise FileNotFoundError(f"Continuous columns info not found at {continuous_cols_path}")
    continuous_cols = joblib.load(continuous_cols_path)

    df, _, _ = scale_data_with_scaler(df, scaler=scaler, fit_scaler=False, continuous_cols=continuous_cols)
    
    save_data(df, csv_out_path)


def post_split_preprocessing(csv_in_path: str, csv_out_path: str) -> None:
    """
    Legacy post-split preprocessing (no scaling to maintain backward compatibility).
    
    NOTE: This function does not perform scaling. Use post_split_preprocessing_train
    and post_split_preprocessing_test for proper handling with scaler persistence.
    
    Args:
        csv_in_path: Path to input CSV.
        csv_out_path: Path to save output CSV.
    """
    df = load_data(csv_in_path)
    # No scaling applied here to prevent leakage
    save_data(df, csv_out_path)


if __name__ == "__main__":
    post_split_preprocessing(str(WIDE_CSV_PATH), str(PREPROCESSED_CSV_PATH))
