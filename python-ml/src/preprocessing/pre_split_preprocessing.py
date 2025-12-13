"""Pre-split preprocessing pipeline including labeling, encoding, and feature selection.

Executes before train/test split to prevent data leakage. Handles:
- Creating future-looking labels based on price movements
- One-hot encoding of categorical features
- Removal of sparse/constant columns
"""
from pathlib import Path
from typing import Tuple
import pandas as pd

from src.config.config import *
from src.preprocessing.feature_engineering import to_feature_engineered
from src.preprocessing.technical_features import add_technical_features
from src.utils.csv_utils import load_csv, save_csv
from src.utils.path_utils import get_skuld_root


def drop_sparse_columns(
    df: pd.DataFrame,
    min_non_zero_ratio: float = 0.01,
    min_unique_values: int = 2
) -> pd.DataFrame:
    """Remove columns that lack sufficient data variation or non-null values.
    
    Preserves special columns (timestamps, labels, one-hot encoded features).
    
    Args:
        df: Input DataFrame.
        min_non_zero_ratio: Minimum ratio of non-zero/non-null values (0-1).
        min_unique_values: Minimum unique values required (removes constants).
    
    Returns:
        pd.DataFrame: DataFrame with sparse columns removed.
    
    Raises:
        ValueError: If min_non_zero_ratio not in [0, 1] or min_unique_values < 1.
    """
    if not (0 <= min_non_zero_ratio <= 1):
        raise ValueError(f"min_non_zero_ratio must be in [0, 1], got {min_non_zero_ratio}")
    
    if min_unique_values < 1:
        raise ValueError(f"min_unique_values must be >= 1, got {min_unique_values}")
    
    # Vectorized approach: compute nunique once, then filter in bulk
    nunique = df.nunique()
    # Only include special columns that actually exist in the dataframe
    special_cols = [col for col in [TIMESTAMP_COL, LABEL_COL, TIMESTAMP_SCALED_COL] if col in df.columns]
    
    # Boolean mask: keep special columns or columns with enough unique values
    is_special = df.columns.isin(special_cols) | df.columns.str.startswith(TICKER_PREFIX)
    has_variance = nunique >= min_unique_values
    keep_mask = is_special | has_variance
    
    cols_to_keep = df.columns[keep_mask].tolist()
    
    # Additional checks for binary and continuous columns with non-zero ratio threshold
    cols_to_check = [col for col in cols_to_keep if col not in special_cols]
    final_keep = [col for col in special_cols if col in cols_to_keep] + [col for col in cols_to_check if col in df.columns]
    
    for col in cols_to_check:
        if col not in df.columns:
            continue
        
        # For binary columns (0/1), check if they have enough positive cases
        unique_vals = set(df[col].unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            # Count non-zero values for binary columns
            non_zero_ratio = (df[col] != 0).sum() / len(df)
            if non_zero_ratio < min_non_zero_ratio:
                if col in final_keep:
                    final_keep.remove(col)
        else:
            # For continuous columns, check for non-null ratio
            non_null_ratio = df[col].notna().sum() / len(df)
            if non_null_ratio < min_non_zero_ratio:
                if col in final_keep:
                    final_keep.remove(col)
    
    cols_to_keep = final_keep

    dropped_cols = set(df.columns) - set(cols_to_keep)

    return df[cols_to_keep]


def create_future_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary labels indicating future price movements.
    
    For each row, determines if the Close price increases by at least THRESHOLD_PCT
    within LABEL_LOOKAHEAD_MILLIS milliseconds.
    
    Label values:
    - 1: Price increased by threshold or more
    - 0: Price did not increase by threshold
    - -1: No valid future data exists
    
    Args:
        df: Input DataFrame with TIMESTAMP_COL, TICKER_COL, and CLOSE_COL.
    
    Returns:
        pd.DataFrame: DataFrame with LABEL_COL column added.
    
    Raises:
        KeyError: If required columns are missing.
        ValueError: If LABEL_LOOKAHEAD_MILLIS is not positive.
    """
    if LABEL_LOOKAHEAD_MILLIS <= 0:
        raise ValueError(f"LABEL_LOOKAHEAD_MILLIS must be positive, got {LABEL_LOOKAHEAD_MILLIS}")
    df = df.sort_values([TICKER_COL, TIMESTAMP_COL]).reset_index(drop=True)
    df["future_ts"] = df[TIMESTAMP_COL] + LABEL_LOOKAHEAD_MILLIS

    labeled_frames = []

    for ticker, group in df.groupby(TICKER_COL):
        g = group.copy()
        # Find index of the first row at or after the future timestamp
        future_idx = g[TIMESTAMP_COL].searchsorted(g["future_ts"], side="left")

        # Compute future prices
        future_prices = [
            g.iloc[idx][CLOSE_COL] if idx < len(g) else None
            for idx in future_idx
        ]
        g["future_close"] = future_prices

        # Compute label: 1 if increase >= threshold, 0 if not, -1 if no valid future data
        g[LABEL_COL] = g.apply(
            lambda row: -1 if pd.isna(row["future_close"])
            else int(((row["future_close"] - row[CLOSE_COL]) / (row[CLOSE_COL] + 0.0000000001)) >= THRESHOLD_PCT),
            axis=1
        ).astype("int8")

        labeled_frames.append(g)

    df_out = pd.concat(labeled_frames, ignore_index=True)

    # Drop helper columns (keep all rows now)
    df_out = df_out.drop(columns=["future_ts", "future_close"])

    return df_out


# =======================================================
# === ONE-HOT ENCODING =================================
# =======================================================

def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the ticker column.
    
    Converts categorical ticker values into binary feature columns
    for use in machine learning models.
    
    Args:
        df: DataFrame with TICKER_COL column.
    
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded ticker columns.
    
    Raises:
        KeyError: If TICKER_COL is missing.
    """
    if TICKER_COL not in df.columns:
        raise KeyError(f"Missing {TICKER_COL} column for one-hot encoding")
    df = pd.get_dummies(df, columns=[TICKER_COL], prefix=TICKER_PREFIX, dtype="int8")
    return df


# =======================================================
# === RESTORE TICKER FOR EVALUATION ====================
# =======================================================

def restore_ticker_delete_one_hot_and_save(input_csv_path: str, output_csv_path: str) -> None:
    """
    Restore the original ticker column and remove one-hot encoded ticker columns.

    Args:
        input_csv_path: Path to CSV with one-hot encoded tickers.
        output_csv_path: Path to save CSV with restored ticker column.
    """
    df = load_csv(input_csv_path)

    # Restore the ticker column from one-hot encoding
    df = restore_ticker_column(df)

    # Find and drop all one-hot encoded ticker columns
    ticker_cols = [c for c in df.columns if c.startswith(f"{TICKER_PREFIX}_")]
    df = df.drop(columns=ticker_cols)

    save_csv(df, output_csv_path)

    print(f"Decoded tickers saved to: {output_csv_path}")
    print(f"Restored ticker column, removed {len(ticker_cols)} one-hot columns")


def restore_ticker_column(df: pd.DataFrame, prefix: str = TICKER_PREFIX) -> pd.DataFrame:
    """Reconstruct the original ticker column from one-hot encoded columns.
    
    Assumes only one 1 per row for the ticker one-hot columns.
    Returns original DataFrame if no one-hot columns found (already restored).
    
    Args:
        df: DataFrame with one-hot encoded ticker columns.
        prefix: Prefix used in one-hot encoding.
    
    Returns:
        pd.DataFrame: DataFrame with restored TICKER_COL column.
    
    Raises:
        ValueError: If restoration fails (ambiguous encoding).
    """
    ticker_cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
    if not ticker_cols:
        # No one-hot columns, assume ticker column exists
        return df

    # Reconstruct ticker
    df[TICKER_COL] = df[ticker_cols].idxmax(axis=1)
    # Remove prefix to restore original ticker names
    df[TICKER_COL] = df[TICKER_COL].str[len(prefix) + 1:]  # +1 for underscore
    return df


# =======================================================
# === FULL PREPROCESSING PIPELINE =====================
# =======================================================

def remove_unlabeled(in_csv_path: str, out_csv_path: str) -> None:
    """Remove rows with invalid labels (-1).
    
    Drops rows where label creation failed (no future price data available).
    
    Args:
        in_csv_path: Path to input CSV.
        out_csv_path: Path to save filtered CSV.
    
    Raises:
        KeyError: If LABEL_COL is missing from input.
        IOError: If file operations fail.
    """
    df = load_csv(in_csv_path)
    
    if LABEL_COL not in df.columns:
        raise KeyError(f"Missing {LABEL_COL} column in {in_csv_path}")
    
    df_valid = df[df[LABEL_COL] != -1].copy()
    save_csv(df_valid, out_csv_path)


def pre_split_preprocess(in_csv_path: str, out_csv_path: str) -> None:
    """Execute complete pre-split preprocessing pipeline.
    
    Steps:
    1. Add technical indicators (momentum, trend, volatility)
    2. Create future labels based on price movements
    3. One-hot encode ticker column
    4. Drop sparse columns with insufficient data variation
    5. Save preprocessed data
    
    Args:
        in_csv_path: Path to input wide-format CSV.
        out_csv_path: Path to save preprocessed CSV.
    
    Raises:
        FileNotFoundError: If input CSV not found.
        ValueError: If preprocessing steps fail.
    """
    if not Path(in_csv_path).exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv_path}")
    df = load_csv(in_csv_path)
    
    # Add technical features to capture momentum, trend, and volatility
    df = add_technical_features(df)
    
    df = create_future_labels(df)
    df = one_hot_encode(df)
    df = drop_sparse_columns(df)
    save_csv(df, out_csv_path)


# =======================================================
# === ENTRYPOINT =======================================
# =======================================================


if __name__ == "__main__":
    pre_split_preprocess(str(WIDE_CSV_PATH), str(PREPROCESSED_CSV_PATH))
