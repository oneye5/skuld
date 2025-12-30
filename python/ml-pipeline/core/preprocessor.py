"""Module for data preprocessing (cleaning, NaN handling)."""

import pandas as pd
import numpy as np
import warnings

from config.columns import TIMESTAMP, TICKER, TARGET
from config.settings import CLIP_THRESHOLD


def preprocess_data(df: pd.DataFrame, add_missing_flags: bool = True) -> pd.DataFrame:
    """Preprocess data: handle NaN, infinities, and type conversions.
    
    Following nzx-predictor Java approach exactly:
    - Replace infinities with NaN
    - Add MissingFlag columns (1=present, 0=missing) for ALL numeric features
    - Fill NaN with 0.0
    - Convert booleans to int
    
    The missing flag pattern matches CsvWriter.java:
    - Flag = 1 means data was present (observed)
    - Flag = 0 means data was missing (imputed)
    
    IMPORTANT: Forward fill is applied per-ticker in TIMESTAMP order to prevent
    future data from leaking into past observations.
    
    Args:
        df: DataFrame to preprocess.
        add_missing_flags: If True, add binary flag columns for ALL features.
            This matches nzx-predictor's approach exactly.
    
    Returns:
        Preprocessed DataFrame.
    
    Raises:
        ValueError: If DataFrame is empty or missing required columns.
    """
    if df.empty:
        warnings.warn("preprocess_data received empty DataFrame", UserWarning)
        return df
    
    result = df.copy()
    
    # Remove any unnamed/index columns
    result = result.loc[:, ~result.columns.str.contains("^Unnamed")]
    
    # Convert booleans to integers
    bool_cols = result.select_dtypes(include="bool").columns
    result[bool_cols] = result[bool_cols].astype(int)
    
    # Get numeric columns (excluding metadata)
    excluded = [TIMESTAMP, TICKER, TARGET]
    numeric_cols = [
        c for c in result.columns 
        if c not in excluded and pd.api.types.is_numeric_dtype(result[c])
    ]
    
    # Replace infinities with NaN first
    for col in numeric_cols:
        result[col] = result[col].replace([np.inf, -np.inf], np.nan)
    
    # Forward fill missing values within each ticker group
    # CRITICAL: Sort by TICKER then TIMESTAMP to ensure forward fill only
    # propagates PAST values to PRESENT (not future to past - would cause leakage)
    if TICKER in result.columns and TIMESTAMP in result.columns:
        # Sort to ensure correct temporal ordering within each ticker
        result = result.sort_values([TICKER, TIMESTAMP]).reset_index(drop=True)
        
        # Forward fill numeric columns by ticker
        # This propagates the last known value (e.g. yesterday's price/rate)
        result[numeric_cols] = result.groupby(TICKER)[numeric_cols].ffill()
    elif TIMESTAMP in result.columns:
        # No ticker column - sort by timestamp only
        result = result.sort_values(TIMESTAMP).reset_index(drop=True)
        result[numeric_cols] = result[numeric_cols].ffill()

    # Add missing flags for ALL features (matching nzx-predictor Java)
    # Flag = 1 means present, Flag = 0 means missing
    if add_missing_flags:
        missing_flag_cols = {}
        for col in numeric_cols:
            # Create flag: 1 if NOT NaN (present), 0 if NaN (missing)
            present_mask = result[col].notna()
            missing_flag_cols[f"MissingFlag_{col}"] = present_mask.astype(np.uint8)
        
        # Add all missing flags at once (more efficient)
        if missing_flag_cols:
            flag_df = pd.DataFrame(missing_flag_cols, index=result.index)
            result = pd.concat([result, flag_df], axis=1)
    
    # Fill NaN with 0.0
    for col in numeric_cols:
        result[col] = result[col].fillna(0.0)
    
    return result


def clip_extreme_values(
    df: pd.DataFrame,
    threshold: float = CLIP_THRESHOLD,
) -> pd.DataFrame:
    """Clip extreme values in scaled features.
    
    After RobustScaler, most values should be in a reasonable range,
    but extreme outliers can still exist. This clips them to avoid
    model instability.
    
    Args:
        df: DataFrame with scaled features.
        threshold: Values beyond [-threshold, threshold] are clipped.
    
    Returns:
        DataFrame with clipped values.
    """
    result = df.copy()
    
    # Columns to exclude from clipping
    excluded = [TIMESTAMP, TICKER, TARGET]
    
    for col in result.columns:
        if col in excluded:
            continue
        if "MissingFlag" in col:
            continue  # Binary flags, don't clip
        if not pd.api.types.is_numeric_dtype(result[col]):
            continue
        
        # Clip to [-threshold, threshold]
        result[col] = result[col].clip(-threshold, threshold)
    
    return result


def drop_sparse_columns(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Drop columns with too many missing values.
    
    Args:
        df: DataFrame to filter.
        threshold: Maximum fraction of missing values allowed.
    
    Returns:
        DataFrame with sparse columns removed.
    """
    missing_frac = df.isnull().mean()
    cols_to_keep = missing_frac[missing_frac < threshold].index.tolist()
    
    return df[cols_to_keep]
