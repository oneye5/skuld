from operator import contains

import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.config.config import *


def min_max_scale_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    global_min = df[TIMESTAMP_COL].min()
    global_max = df[TIMESTAMP_COL].max()
    df[TIMESTAMP_SCALED_COL] = (df[TIMESTAMP_COL] - global_min) / (global_max - global_min)
    return df


def drop_sparse_columns(df: pd.DataFrame, min_non_zero_ratio: float = 0.01, min_unique_values: int = 2) -> pd.DataFrame:
    cols_to_keep = []

    for col in df.columns:
        # Always keep timestamp and label columns
        if col in [TIMESTAMP_COL, LABEL_COL, TIMESTAMP_SCALED_COL] or col.__contains__(TICKER_PREFIX):
            cols_to_keep.append(col)
            continue

        # Check if column has minimum unique values (remove constants)
        if df[col].nunique() < min_unique_values:
            continue

        # For binary columns (0/1), check if they have enough positive cases
        unique_vals = set(df[col].unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            # Count non-zero values for binary columns
            non_zero_ratio = (df[col] != 0).sum() / len(df)
            if non_zero_ratio >= min_non_zero_ratio:
                cols_to_keep.append(col)
        else:
            # For continuous columns, check for non-null ratio
            non_null_ratio = df[col].notna().sum() / len(df)
            if non_null_ratio >= min_non_zero_ratio:
                cols_to_keep.append(col)

    dropped_cols = set(df.columns) - set(cols_to_keep)
    if dropped_cols:
        print(f"Dropped {len(dropped_cols)} sparse columns: {sorted(dropped_cols)}")

    return df[cols_to_keep]

def scale_data(df: pd.DataFrame) -> pd.DataFrame:
    true_binary_cols = []
    for col in df.columns:
        # 1. Check if the column has exactly two unique values
        if df[col].nunique() == 2:
            # 2. Check if those two unique values are 0 and 1 (allowing for floating point comparison)
            unique_vals = set(df[col].unique())
            if unique_vals == {0, 1} or unique_vals == {0.0, 1.0}:
                true_binary_cols.append(col)

    # Define continuous columns: all columns that are NOT ID/LABEL/TIMESTAMP
    # and are NOT one of the strictly defined binary columns.
    continuous_cols = [
        col for col in df.columns
        if col not in true_binary_cols
           and col != TIMESTAMP_COL
           and col != LABEL_COL
           and col != CLOSE_COL
    ]

    # 1. Ensure columns are floating point to prevent truncation (as discussed previously)
    df[continuous_cols] = df[continuous_cols].astype(float)

    # 2. Apply scaling
    scaler = RobustScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    return df

def to_feature_engineered(df: pd.DataFrame) -> pd.DataFrame:
    if FE_ENABLE_DROP_SPARSE_COLUMNS:
        df = drop_sparse_columns(df)
    if FE_ENABLE_SCALE_WHOLE_DATASET:
        df = min_max_scale_time(df)
    if FE_ENABLE_MIN_MAX_SCALE_TIME_COLUMN:
        df = scale_data(df)
    return df