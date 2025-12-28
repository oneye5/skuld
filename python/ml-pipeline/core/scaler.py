"""Module for scaling features."""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

from config.columns import TIMESTAMP, TICKER, TARGET, TIME_SCALED
from core.target_builder import FORWARD_RETURN


@dataclass
class ScalerSet:
    """Container for fitted scalers."""
    continuous_scaler: RobustScaler
    continuous_cols: list[str]
    binary_cols: list[str]
    excluded_cols: list[str]


def fit_scaler(df: pd.DataFrame) -> ScalerSet:
    """Fit scalers on the data.
    
    Uses RobustScaler for continuous features (robust to outliers).
    Binary features are not scaled.
    
    Following nzx-predictor approach: fit on combined train+test data.
    
    Args:
        df: DataFrame to fit scalers on.
    
    Returns:
        ScalerSet with fitted scaler and column information.
    """
    # Columns to exclude from scaling
    # TIME_SCALED is already 0-1 normalized, don't scale it again
    # FORWARD_RETURN is the target for ranking - must not be scaled
    excluded_cols = [TIMESTAMP, TICKER, TARGET, TIME_SCALED, FORWARD_RETURN]
    excluded_cols = [c for c in excluded_cols if c in df.columns]
    
    # Identify binary columns (exactly 2 unique values or all 0/1)
    binary_cols = []
    continuous_cols = []
    
    for col in df.columns:
        if col in excluded_cols:
            continue
        
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            excluded_cols.append(col)
            continue
        
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 2:
            binary_cols.append(col)
        else:
            continuous_cols.append(col)
    
    # Fit RobustScaler on continuous columns
    scaler = RobustScaler()
    if continuous_cols:
        scaler.fit(df[continuous_cols].values)
    
    return ScalerSet(
        continuous_scaler=scaler,
        continuous_cols=continuous_cols,
        binary_cols=binary_cols,
        excluded_cols=excluded_cols,
    )


def transform_data(df: pd.DataFrame, scaler_set: ScalerSet) -> pd.DataFrame:
    """Transform data using fitted scalers.
    
    Args:
        df: DataFrame to transform.
        scaler_set: Fitted ScalerSet from fit_scaler().
    
    Returns:
        Transformed DataFrame.
    """
    result = df.copy()
    
    if scaler_set.continuous_cols:
        # Only transform columns that exist in this DataFrame
        cols_to_transform = [c for c in scaler_set.continuous_cols if c in result.columns]
        
        if cols_to_transform:
            # If all fitted columns exist, use the scaler directly
            if cols_to_transform == scaler_set.continuous_cols:
                transformed = scaler_set.continuous_scaler.transform(
                    result[cols_to_transform].values
                ).astype(np.float32)
                result[cols_to_transform] = transformed
            else:
                # Only some columns exist - transform column by column using the 
                # fitted parameters for each column
                for col in cols_to_transform:
                    col_idx = scaler_set.continuous_cols.index(col)
                    center = scaler_set.continuous_scaler.center_[col_idx]
                    scale = scaler_set.continuous_scaler.scale_[col_idx]
                    result[col] = ((result[col].values - center) / scale).astype(np.float32)
    
    return result
