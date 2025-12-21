"""Module for handling missing data with imputation and indicator columns."""

from dataclasses import dataclass
import pandas as pd
import numpy as np

from config.column_names import TIMESTAMP, TICKER, TARGET, IMPUTED_PREFIX, MISSING_PREFIX


@dataclass
class ImputationStats:
    """Statistics computed from training data for imputation."""
    feature_means: dict[str, float]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get feature columns (excluding metadata columns)."""
    exclude = {TIMESTAMP, TICKER, TARGET}
    return [col for col in df.columns if col not in exclude]


def compute_imputation_stats(df: pd.DataFrame) -> ImputationStats:
    """
    Compute imputation statistics from training data.
    
    Args:
        df: Training DataFrame to compute statistics from.
    
    Returns:
        ImputationStats containing feature means for imputation.
    """
    feature_cols = get_feature_columns(df)
    feature_means = {}
    
    for col in feature_cols:
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            mean_val = df[col].mean()
            # Use 0.0 for columns that are 100% NaN (they'll be dropped by feature selection)
            feature_means[col] = 0.0 if pd.isna(mean_val) else mean_val
    
    return ImputationStats(feature_means=feature_means)


def impute_data(
    df: pd.DataFrame,
    stats: ImputationStats,
    add_indicators: bool = True,
) -> pd.DataFrame:
    """
    Impute missing values and optionally add indicator columns.
    
    Uses forward fill within each ticker (to avoid future data leakage),
    then fills remaining NaN with training set means.
    
    Args:
        df: DataFrame to impute.
        stats: ImputationStats computed from training data.
        add_indicators: Whether to add missing/imputed indicator columns.
    
    Returns:
        DataFrame with imputed values and optional indicator columns.
    """
    # Sort by ticker and timestamp for proper forward fill (inplace)
    df.sort_values([TICKER, TIMESTAMP], inplace=True)
    feature_cols = [col for col in get_feature_columns(df) if col in stats.feature_means]
    
    # Compute missing indicators before imputation (batch operation)
    if add_indicators:
        missing_data = {f"{MISSING_PREFIX}{col}": df[col].isna().astype('int8') for col in feature_cols}
    
    # Forward fill within each ticker (no future data leakage) - batch operation
    df[feature_cols] = df.groupby(TICKER, sort=False)[feature_cols].ffill()
    
    # Fill remaining NaN with training means - batch operation
    for col in feature_cols:
        df[col] = df[col].fillna(stats.feature_means[col])
    
    # Add indicator columns at the end using concat (avoids fragmentation)
    if add_indicators:
        # Create imputed indicators (same as missing since we filled everything) using int8 dtype
        imputed_data = {f"{IMPUTED_PREFIX}{col}": missing_data[f"{MISSING_PREFIX}{col}"] for col in feature_cols}
        
        # Combine all indicator columns and concat at once with efficient dtype
        indicator_df = pd.DataFrame({**missing_data, **imputed_data}, index=df.index, dtype='int8')
        df = pd.concat([df, indicator_df], axis=1, copy=False)
    
    return df
