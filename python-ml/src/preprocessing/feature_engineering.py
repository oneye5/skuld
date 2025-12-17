"""Feature engineering and data scaling operations.

Provides RobustScaler-based scaling that:
- Prevents data leakage by fitting only on training data
- Preserves binary and categorical columns
- Validates feature alignment between train/test
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Optional, List, Tuple

from src.config.config import *


def scale_data_with_scaler(
    df: pd.DataFrame,
    scaler: Optional[RobustScaler] = None,
    fit_scaler: bool = False,
    continuous_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Optional[RobustScaler], List[str]]:
    """Scale continuous features using RobustScaler (resistant to outliers).
    
    Preserves binary columns (0/1) and non-numeric columns unchanged.
    Can either fit a new scaler (training) or apply existing one (testing).
    
    Args:
        df: Input DataFrame.
        scaler: Pre-fitted RobustScaler. Required if fit_scaler=False.
        fit_scaler: Whether to fit scaler on this data (training mode).
                   If False, applies provided scaler (test mode).
        continuous_cols: List of columns to scale. If None, infers from data.
    
    Returns:
        Tuple[pd.DataFrame, RobustScaler, List[str]]: Scaled data, fitted scaler,
        and list of columns that were scaled.
    
    Raises:
        ValueError: If fit_scaler=False but scaler is None, or if column count mismatch.
    """
    if not fit_scaler and scaler is None:
        raise ValueError("Must provide fitted scaler when fit_scaler=False")
    
    df = df.copy()
    
    # Identify binary columns (only scale if explicitly numeric)
    true_binary_cols = []
    for col in df.columns:
        if df[col].dtype in ['int8', 'uint8', 'int16', 'uint16']:
            try:
                unique_vals = set(df[col].dropna().unique())
                if len(unique_vals) <= 2 and unique_vals.issubset({0, 1}):
                    true_binary_cols.append(col)
            except:
                pass

    # Define continuous columns: exclude special columns and binary features
    inferred_continuous_cols = [
        col for col in df.columns
        if col not in true_binary_cols
        and col not in [TIMESTAMP_COL, LABEL_COL, CLOSE_COL, TIMESTAMP_SCALED_COL]
        and not col.startswith(TICKER_PREFIX)
        and df[col].dtype in ['int32', 'int64', 'float32', 'float64']
    ]

    # Use provided columns or inferred ones
    if continuous_cols is not None:
        missing = [c for c in continuous_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Provided columns missing from DataFrame: {missing}")
        continuous_cols = list(continuous_cols)
    else:
        continuous_cols = inferred_continuous_cols

    if not continuous_cols:
        return df, scaler, continuous_cols

    # Ensure floating point
    df[continuous_cols] = df[continuous_cols].astype(float)

    if fit_scaler:
        scaler = RobustScaler()
        df[continuous_cols] = scaler.fit_transform(df[continuous_cols].values)
    else:
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != len(continuous_cols):
            raise ValueError(
                f"Scaler expects {scaler.n_features_in_} features, got {len(continuous_cols)}"
            )
        df[continuous_cols] = scaler.transform(df[continuous_cols].values)

    return df, scaler, continuous_cols


def scale_continuous_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale continuous features using RobustScaler.
    
    WARNING: Only use this before train/test split. For proper handling
    after split, use scale_data_with_scaler() with fit_scaler control.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with scaled continuous features.
    """
    df_scaled, _, _ = scale_data_with_scaler(df, scaler=None, fit_scaler=True)
    return df_scaled


def to_feature_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Apply pre-split feature engineering transformations.
    
    WARNING: Scaling is NOT applied here to prevent data leakage.
    Scaling happens separately in post_split_preprocessing.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        pd.DataFrame: Feature engineered DataFrame (no scaling).
    """
    return df