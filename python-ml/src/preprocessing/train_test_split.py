"""Train-test splitting utilities for time-based data splits."""
from pathlib import Path
import pandas as pd

from src.config.config import *
from src.preprocessing.pre_split_preprocessing import restore_ticker_column
from src.utils.io_utils import load_data, save_data
from src.utils.data_validation import validate_time_series_integrity, print_data_quality_report
from src.utils.path_utils import get_skuld_root


def time_based_split(
    df: pd.DataFrame,
    from_ts: int,
    to_ts: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into train and test sets based on a specific time window.
    
    IMPORTANT: This ensures no temporal leakage by enforcing train_max_ts < test_min_ts.

    Args:
        df: Input dataframe with TIMESTAMP_COL column.
        from_ts: Timestamp marking the end of Train and start of Test.
        to_ts: Timestamp marking the end of Test. Data after this is dropped.

    Returns:
        tuple: (train_df, test_df)
        
    Raises:
        ValueError: If split results in empty train or test set.
    """
    # Ensure sorted by timestamp
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # Train: Everything BEFORE the 'from' timestamp
    train_df = df[df[TIMESTAMP_COL] < from_ts].copy()

    # Test: Everything FROM 'from' UP TO 'to'
    # Data occurring after 'to_ts' is implicitly dropped by not being included here
    test_df = df[(df[TIMESTAMP_COL] >= from_ts) & (df[TIMESTAMP_COL] <= to_ts)].copy()

    # Validate split
    if train_df.empty:
        raise ValueError(f"Training set is empty for time range < {from_ts}")
    if test_df.empty:
        raise ValueError(f"Test set is empty for time range [{from_ts}, {to_ts}]")
    
    # Verify no temporal leakage
    validation = validate_time_series_integrity(train_df, test_df)
    if not validation['no_temporal_overlap']:
        raise ValueError("LEAKAGE DETECTED: Train and test sets have temporal overlap!")
    
    return train_df, test_df


def split_last_occurring_tickers(
    preprocessed_csv_path: str,
    train_csv_path: str,
    last_rows_csv_path: str
) -> None:
    """
    Split data by extracting the last occurrence of each ticker.
    
    The last occurring rows for each ticker (with labels stripped) are saved separately.
    Remaining data has invalid labels (-1) removed and is saved for training.
    
    Args:
        preprocessed_csv_path: Path to preprocessed CSV with invalid labels.
        train_csv_path: Path to save training data.
        last_rows_csv_path: Path to save last rows (label column removed).
    """
    df = load_data(preprocessed_csv_path)

    # Restore the ticker column from one-hot encoding (as helper column)
    df = restore_ticker_column(df)

    # Sort by timestamp to ensure we get the truly last occurring rows
    df = df.sort_values([TICKER_COL, TIMESTAMP_COL]).reset_index(drop=True)

    # Get the last row for each ticker (already sorted, use sort=False to avoid re-sort)
    last_rows = df.groupby(TICKER_COL, sort=False).tail(1).drop(columns=[LABEL_COL, TICKER_COL])

    # Get remaining data (everything except the last rows)
    last_row_indices = last_rows.index
    remaining_df = df.drop(index=last_row_indices)

    # Drop invalid labels (-1) and ticker column from remaining data
    train_df = remaining_df[remaining_df[LABEL_COL] != -1].drop(columns=[TICKER_COL])

    # Validate split
    if train_df.empty:
        raise ValueError("Training set is empty after filtering invalid labels")
    if last_rows.empty:
        raise ValueError("Test set (last rows) is empty")

    # Save both dataframes
    save_data(last_rows, last_rows_csv_path)
    save_data(train_df, train_csv_path)

    print(f"Last rows saved to: {last_rows_csv_path} ({len(last_rows)} rows)")
    print(f"Train data saved to: {train_csv_path} ({len(train_df)} rows)")
    
    # Print quality reports
    print_data_quality_report(train_df, "Training Data")
    print_data_quality_report(last_rows, "Test Data (Last Rows)")


def split_and_save(preprocessed_csv_path: str, from_ts: int, to_ts: int) -> None:
    """
    Load preprocessed data, split into train/test based on timestamps, and save.
    
    Args:
        preprocessed_csv_path: Path to preprocessed CSV.
        from_ts: Timestamp marking train/test boundary.
        to_ts: Timestamp marking test end.
    """
    df = load_data(preprocessed_csv_path)

    # Perform the split with validation
    train_df, test_df = time_based_split(df, from_ts, to_ts)

    save_data(train_df, str(TRAIN_CSV_PATH))
    save_data(test_df, str(TEST_CSV_PATH))

    print(f"--- Split Complete (No Temporal Leakage) ---")
    print(f"Train Window:  Start -> {from_ts}")
    print(f"Test Window:   {from_ts} -> {to_ts}")
    print(f"Dropped Data:  {to_ts} -> End")
    print(f"----------------------")
    print(f"Train CSV saved to: {TRAIN_CSV_PATH} ({len(train_df)} rows)")
    print(f"Test CSV saved to:  {TEST_CSV_PATH} ({len(test_df)} rows)")
    
    # Print quality reports
    print_data_quality_report(train_df, "Training Data")
    print_data_quality_report(test_df, "Test Data")


if __name__ == "__main__":
    FROM_TS = 1715000000000 - (1000 * 60 * 60 * 24 * 365)  # Start of Test Data, -1 year
    TO_TS = 1715000000000  # End of Test Data

    print("Loading preprocessed data:", PREPROCESSED_CSV_PATH)
    split_and_save(str(PREPROCESSED_CSV_PATH), FROM_TS, TO_TS)