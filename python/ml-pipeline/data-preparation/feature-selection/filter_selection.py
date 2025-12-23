"""Filter-based feature selection - removing uninformative features.

Filter methods select features based on statistical properties computed
independently of the model. These are fast and model-agnostic.

Methods included:
- Variance threshold: Remove near-constant features
- Correlation threshold: Remove highly correlated features (keep first)
- Missing value threshold: Remove features with too many NaN values
"""

import pandas as pd
import numpy as np
from typing import Tuple

from config.column_names import TIMESTAMP, TICKER, TARGET


# Thresholds for dropping features
VARIANCE_THRESHOLD = 0.01  # Drop if variance below this (after scaling)
CORRELATION_THRESHOLD = 0.95  # Drop if correlation above this with another feature
MISSING_THRESHOLD = 0.50  # Drop if more than 50% missing


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of feature columns (excluding metadata and target)."""
    exclude_cols = {TIMESTAMP, TICKER, TARGET, "index"}
    return [col for col in df.columns if col not in exclude_cols]


def select_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variance_threshold: float = VARIANCE_THRESHOLD,
    correlation_threshold: float = CORRELATION_THRESHOLD,
    missing_threshold: float = MISSING_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Select informative features based on training data statistics.
    
    Applies three filters in order:
    1. Remove features with too many missing values
    2. Remove near-zero variance features
    3. Remove highly correlated features (keeping first)
    
    Args:
        train_df: Training DataFrame (used to compute selection criteria).
        test_df: Test DataFrame (same features will be dropped).
        variance_threshold: Min variance to keep a feature.
        correlation_threshold: Max correlation before dropping one feature.
        missing_threshold: Max missing ratio before dropping.
    
    Returns:
        Tuple of (filtered_train, filtered_test, dropped_features).
    """
    feature_cols = get_feature_columns(train_df)
    
    if not feature_cols:
        return train_df, test_df, []
    
    dropped_features = []
    cols_to_keep = feature_cols.copy()
    
    # 1. Drop features with too many missing values
    cols_to_keep, dropped_missing = _drop_high_missing(
        train_df, cols_to_keep, missing_threshold
    )
    dropped_features.extend(dropped_missing)
    
    # 2. Drop near-zero variance features
    cols_to_keep, dropped_variance = _drop_low_variance(
        train_df, cols_to_keep, variance_threshold
    )
    dropped_features.extend(dropped_variance)
    
    # 3. Drop highly correlated features
    cols_to_keep, dropped_corr = _drop_high_correlation(
        train_df, cols_to_keep, correlation_threshold
    )
    dropped_features.extend(dropped_corr)
    
    # Build final column list (metadata + selected features + target)
    metadata_cols = [col for col in [TIMESTAMP, TICKER] if col in train_df.columns]
    target_cols = [TARGET] if TARGET in train_df.columns else []
    final_cols = metadata_cols + cols_to_keep + target_cols
    
    # Filter both DataFrames
    train_filtered = train_df[final_cols].copy()
    test_filtered = test_df[final_cols].copy()
    
    return train_filtered, test_filtered, dropped_features


def _drop_high_missing(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
) -> Tuple[list[str], list[str]]:
    """Drop features with missing ratio above threshold."""
    dropped = []
    kept = []
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        missing_ratio = df[col].isna().mean()
        if missing_ratio > threshold:
            dropped.append(col)
        else:
            kept.append(col)
    
    return kept, dropped


def _drop_low_variance(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
) -> Tuple[list[str], list[str]]:
    """Drop features with variance below threshold."""
    dropped = []
    kept = []
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        # Compute variance ignoring NaN
        variance = df[col].var(skipna=True)
        
        # Handle case where variance is NaN (all values same or all NaN)
        if pd.isna(variance) or variance < threshold:
            dropped.append(col)
        else:
            kept.append(col)
    
    return kept, dropped


def _drop_high_correlation(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
) -> Tuple[list[str], list[str]]:
    """
    Drop features with correlation above threshold.
    
    When two features are highly correlated, drops the second one
    (keeps the first in the list).
    """
    if len(feature_cols) < 2:
        return feature_cols, []
    
    # Extract feature matrix
    feature_df = df[feature_cols].copy()
    
    # Compute correlation matrix
    corr_matrix = feature_df.corr(method='pearson').abs()
    
    # Find features to drop
    dropped = set()
    kept_cols = list(feature_cols)
    
    for i, col1 in enumerate(feature_cols):
        if col1 in dropped:
            continue
        for col2 in feature_cols[i + 1:]:
            if col2 in dropped:
                continue
            
            # Check correlation
            if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
                corr_val = corr_matrix.loc[col1, col2]
                if not pd.isna(corr_val) and corr_val > threshold:
                    dropped.add(col2)
    
    kept = [col for col in feature_cols if col not in dropped]
    
    return kept, list(dropped)


def compute_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for each feature (useful for analysis).
    
    Returns DataFrame with variance, missing ratio, and other stats.
    """
    feature_cols = get_feature_columns(df)
    
    stats = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        col_data = df[col]
        stats.append({
            'feature': col,
            'variance': col_data.var(skipna=True),
            'std': col_data.std(skipna=True),
            'missing_ratio': col_data.isna().mean(),
            'unique_count': col_data.nunique(),
            'min': col_data.min(),
            'max': col_data.max(),
            'mean': col_data.mean(),
        })
    
    return pd.DataFrame(stats)
