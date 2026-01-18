"""Module for scaling features.

Memory-optimized implementation:
- Uses float32 instead of float64 to halve memory usage
- Processes data in chunks for large datasets
- Avoids creating full DataFrame copies
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

from config.columns import TIMESTAMP, TICKER, TARGET
from core.target_builder import FORWARD_RETURN


# Memory optimization constants
CHUNK_SIZE = 100_000  # Rows per chunk for transformation
USE_FLOAT32 = True    # Use float32 for ~50% memory savings


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
    
    Fits ONLY on the provided DataFrame (typically training data) to avoid leakage.
    
    Memory optimization: Fits on a sample if data is very large.
    
    Args:
        df: DataFrame to fit scalers on.
    
    Returns:
        ScalerSet with fitted scaler and column information.
    """
    # Columns to exclude from scaling
    # FORWARD_RETURN is the target for ranking - must not be scaled
    excluded_cols = [TIMESTAMP, TICKER, TARGET, FORWARD_RETURN]
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
    # For large datasets, sample to reduce memory during fit
    scaler = RobustScaler()
    if continuous_cols:
        n_rows = len(df)
        max_fit_samples = 100_000  # RobustScaler needs only sample for quantiles
        
        if n_rows > max_fit_samples:
            # Sample for fitting - RobustScaler uses quantiles so sampling is fine
            sample_idx = np.random.choice(n_rows, max_fit_samples, replace=False)
            fit_data = df.iloc[sample_idx][continuous_cols].values.astype(np.float32)
        else:
            fit_data = df[continuous_cols].values.astype(np.float32)
        
        scaler.fit(fit_data)
        del fit_data  # Free memory immediately
    
    return ScalerSet(
        continuous_scaler=scaler,
        continuous_cols=continuous_cols,
        binary_cols=binary_cols,
        excluded_cols=excluded_cols,
    )


def transform_data(
    df: pd.DataFrame, 
    scaler_set: ScalerSet,
    strict: bool = False,
) -> pd.DataFrame:
    """Transform data using fitted scalers.
    
    Memory-optimized: Uses chunked processing for large datasets and float32.
    
    Args:
        df: DataFrame to transform.
        scaler_set: Fitted ScalerSet from fit_scaler().
        strict: If True, raise error when required columns are missing.
                Use strict=True for prediction to ensure feature alignment.
                Use strict=False for training pipelines with dynamic features.
    
    Returns:
        Transformed DataFrame.
    
    Raises:
        ValueError: If strict=True and required columns are missing.
    """
    # Avoid full copy - modify in place where possible
    result = df
    
    if scaler_set.continuous_cols:
        # Check which columns exist
        cols_to_transform = [c for c in scaler_set.continuous_cols if c in result.columns]
        missing_cols = set(scaler_set.continuous_cols) - set(result.columns)
        
        # In strict mode, all fitted columns must be present
        if strict and missing_cols:
            raise ValueError(
                f"Strict mode: {len(missing_cols)} columns required by scaler are missing.\\n"
                f"Missing columns: {sorted(missing_cols)[:20]}{'...' if len(missing_cols) > 20 else ''}\\n"
                "This causes feature misalignment and invalid predictions.\\n"
                "Either retrain the model or ensure all required features are present."
            )
        
        if cols_to_transform:
            n_rows = len(result)
            
            # For large datasets, process in chunks to limit memory
            if n_rows > CHUNK_SIZE and cols_to_transform == scaler_set.continuous_cols:
                # Chunked transformation - more memory efficient
                result = df.copy()  # Now we need the copy for in-place modification
                
                for start_idx in range(0, n_rows, CHUNK_SIZE):
                    end_idx = min(start_idx + CHUNK_SIZE, n_rows)
                    chunk_data = result.iloc[start_idx:end_idx][cols_to_transform].values.astype(np.float32)
                    transformed_chunk = scaler_set.continuous_scaler.transform(chunk_data).astype(np.float32)
                    result.iloc[start_idx:end_idx, result.columns.get_indexer(cols_to_transform)] = transformed_chunk
                    del chunk_data, transformed_chunk  # Free chunk memory immediately
                    
            elif cols_to_transform == scaler_set.continuous_cols:
                # All columns present - use the scaler directly
                result = df.copy()
                transformed = scaler_set.continuous_scaler.transform(
                    result[cols_to_transform].values.astype(np.float32)
                ).astype(np.float32)
                result[cols_to_transform] = transformed
                del transformed
            else:
                # Only some columns exist - transform column by column using the 
                # fitted parameters for each column
                # WARNING: This path should rarely be used - indicates potential issues
                result = df.copy()
                import warnings
                warnings.warn(
                    f"Partial column transform: {len(missing_cols)} columns missing from scaler. "
                    "This may cause feature misalignment.",
                    UserWarning
                )
                for col in cols_to_transform:
                    col_idx = scaler_set.continuous_cols.index(col)
                    center = scaler_set.continuous_scaler.center_[col_idx]
                    scale = scaler_set.continuous_scaler.scale_[col_idx]
                    result[col] = ((result[col].values - center) / scale).astype(np.float32)
    
    return result
