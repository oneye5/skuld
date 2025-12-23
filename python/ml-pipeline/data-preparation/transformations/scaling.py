"""Module for scaling features globally using RobustScaler.

Matches the nzx-predictor approach: all continuous features are scaled together
using RobustScaler (robust to outliers), fitted on the entire dataset.
"""

from dataclasses import dataclass
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

from config.column_names import TIMESTAMP, TICKER, TARGET, MACRO_PREFIX


@dataclass
class ScalerSet:
    """Container for fitted scaler."""
    scaler: RobustScaler | None
    continuous_columns: list[str]


def get_continuous_columns(df: pd.DataFrame) -> list[str]:
    """Get all continuous columns to scale (excludes binary, identifiers, target)."""
    exclude = {TIMESTAMP, TICKER, TARGET}
    
    continuous = []
    for col in df.columns:
        # Skip excluded columns
        if col in exclude:
            continue
        # Skip one-hot encoded ticker columns
        if col.startswith("Ticker_"):
            continue
        # Only include numeric columns
        if df[col].dtype not in ['float64', 'int64', 'float32', 'int32']:
            continue
        # Skip binary columns (only 2 unique values)
        if df[col].nunique() <= 2:
            continue
        continuous.append(col)
    
    return continuous


def fit_scalers(df: pd.DataFrame) -> ScalerSet:
    """
    Fit RobustScaler on the data globally (all continuous columns together).
    
    Following nzx-predictor approach: scale all features together, not per-ticker.
    
    Args:
        df: DataFrame with features to scale (can be train+test combined).
    
    Returns:
        ScalerSet containing fitted scaler.
    """
    continuous_cols = get_continuous_columns(df)
    
    scaler = None
    if continuous_cols:
        data = df[continuous_cols].values
        # Handle NaN - fit on non-NaN rows only
        valid_mask = ~np.isnan(data).any(axis=1)
        if valid_mask.any():
            scaler = RobustScaler()
            scaler.fit(data[valid_mask])
    
    return ScalerSet(
        scaler=scaler,
        continuous_columns=continuous_cols,
    )


def transform_data(df: pd.DataFrame, scaler_set: ScalerSet) -> pd.DataFrame:
    """
    Transform data using fitted scaler. Note: Modifies df inplace.
    
    Args:
        df: DataFrame with features to scale.
        scaler_set: ScalerSet containing fitted scaler.
    
    Returns:
        DataFrame with scaled features (same object as input).
    """
    # Get columns that exist in this DataFrame
    cols = [c for c in scaler_set.continuous_columns if c in df.columns]
    
    if scaler_set.scaler is not None and cols:
        data = df[cols].values
        # Handle rows with NaN
        valid_mask = ~np.isnan(data).any(axis=1)
        if valid_mask.any():
            scaled = np.full_like(data, np.nan)
            scaled[valid_mask] = scaler_set.scaler.transform(data[valid_mask])
            # Cast to float32 to match DataFrame dtype
            df[cols] = scaled.astype('float32')
    
    return df


def save_scalers(scaler_set: ScalerSet, output_dir: Path, window_id: int) -> None:
    """Save scaler to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if scaler_set.scaler is not None:
        path = output_dir / f"scaler_window{window_id}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(scaler_set, f)


def load_scalers(output_dir: Path, window_id: int) -> ScalerSet | None:
    """Load scaler from disk."""
    path = output_dir / f"scaler_window{window_id}.pkl"
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None
