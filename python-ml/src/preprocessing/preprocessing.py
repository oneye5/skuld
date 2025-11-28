from pathlib import Path
import pandas as pd

from src.config.config import *
from src.preprocessing.feature_engineering import min_max_scale_time, to_feature_engineered
from src.tests.utils import print_sample_data
from src.utils.csv_utils import load_csv, save_csv
from src.utils.path_utils import get_skuld_root


# =======================================================
# === LABEL GENERATION =================================
# =======================================================

def create_future_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, create a label indicating whether the Close price increases
    by at least THRESHOLD_PCT within FUTURE_DELTA_MILLIS milliseconds.
    The label is 1 if it increases, 0 if it doesn't, and -1 if no valid future data exists.
    """
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
    """
    One-hot encode the ticker column using TICKER_PREFIX from config.
    """
    df = pd.get_dummies(df, columns=[TICKER_COL], prefix=TICKER_PREFIX, dtype="int8")
    return df


# =======================================================
# === RESTORE TICKER FOR EVALUATION ====================
# =======================================================

def restore_ticker_delete_one_hot_and_save(input_csv_path: str, output_csv_path: str):
    """
    Restore the original ticker column and remove one-hot encoded ticker columns.

    Args:
        input_csv_path: Path to CSV with one-hot encoded tickers
        output_csv_path: Path to save CSV with restored ticker column
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
    """
    Reconstruct the original ticker column from one-hot encoded columns.
    Assumes only one 1 per row for the ticker one-hot columns.
    """
    ticker_cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
    if not ticker_cols:
        # No one-hot columns, assume ticker column exists
        return df

    # Reconstruct ticker
    df[TICKER_COL] = df[ticker_cols].idxmax(axis=1)
    # Remove prefix to restore original ticker names
    df[TICKER_COL] = df[TICKER_COL].str[len(prefix)+1:]  # +1 for underscore
    return df


# =======================================================
# === FULL PREPROCESSING PIPELINE =====================
# =======================================================

def remove_unlabeled(in_csv_path: str, out_csv_path: str):
    df = load_csv(in_csv_path)
    df_valid = df[df[LABEL_COL] != -1].copy()
    save_csv(df_valid, out_csv_path)

def preprocess(wide_csv_path: str, output_csv_path: str):
    """
    Full preprocessing pipeline:
    - Load wide CSV
    - Generate future labels (1/0)
    - One-hot encode tickers
    - Save preprocessed CSV
    """
    df = load_csv(wide_csv_path)
    print("Raw wide data")
    print_sample_data(df)

    df = create_future_labels(df)
    print("Labeled data")
    print_sample_data(df)

    df = one_hot_encode(df)
    print("One hot data")
    print_sample_data(df)

    df = to_feature_engineered(df)
    print("Feature engineered")
    print_sample_data(df)

    save_csv(df, output_csv_path)
    print(f"Preprocessed CSV saved to {output_csv_path}")


# =======================================================
# === ENTRYPOINT =======================================
# =======================================================


if __name__ == "__main__":
    print("Loading:", WIDE_CSV_PATH)
    print("Saving:", PREPROCESSED_CSV_PATH)

    preprocess(str(WIDE_CSV_PATH), str(PREPROCESSED_CSV_PATH))
