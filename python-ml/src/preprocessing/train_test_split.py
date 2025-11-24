from pathlib import Path
import pandas as pd

from src.config.config import *
from src.preprocessing.preprocessing import restore_ticker_column
from src.utils.csv_utils import load_csv, save_csv
from src.utils.path_utils import get_skuld_root


def time_based_split(df: pd.DataFrame, from_ts: int, to_ts: int):
    """
    Split the dataframe into train and test sets based on a specific time window.

    Args:
        df: Input dataframe.
        from_ts: The timestamp marking the end of Train and start of Test.
        to_ts: The timestamp marking the end of Test. Data after this is dropped.

    Returns:
        train_df, test_df
    """
    # Ensure sorted by timestamp
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # Train: Everything BEFORE the 'from' timestamp
    train_df = df[df[TIMESTAMP_COL] < from_ts].copy()

    # Test: Everything FROM 'from' UP TO 'to'
    # Data occuring after 'to_ts' is implicitly dropped by not being included here
    test_df = df[(df[TIMESTAMP_COL] >= from_ts) & (df[TIMESTAMP_COL] <= to_ts)].copy()

    return train_df, test_df


def split_last_occurring_tickers(preprocessed_csv_path: str, train_csv_path: str, last_rows_csv_path: str):
    """
    Expects invalid labels
    Selects the last occurring rows that contain unique tickers and saves them to last_rows_csv_path with the label column stripped
    Remaining data first has invalid labels dropped and then saved to train_csv_path
    Invalid labels have values of -1
    """
    df = load_csv(preprocessed_csv_path)

    # Restore the ticker column from one-hot encoding (as helper column)
    df = restore_ticker_column(df)

    # Sort by timestamp to ensure we get the truly last occurring rows
    df = df.sort_values([TICKER_COL, TIMESTAMP_COL]).reset_index(drop=True)

    # Get the last row for each ticker
    last_rows = df.groupby(TICKER_COL).tail(1).copy()

    # Strip the label column from last rows
    last_rows = last_rows.drop(columns=[LABEL_COL])

    # Drop the ticker column (non-numeric helper column)
    last_rows = last_rows.drop(columns=[TICKER_COL])

    # Get remaining data (everything except the last rows)
    last_row_indices = last_rows.index
    remaining_df = df.drop(index=last_row_indices).copy()

    # Drop invalid labels (-1) from remaining data
    train_df = remaining_df[remaining_df[LABEL_COL] != -1].copy()

    # Drop the ticker column from train data as well
    train_df = train_df.drop(columns=[TICKER_COL])

    # Save both dataframes
    save_csv(last_rows, last_rows_csv_path)
    save_csv(train_df, train_csv_path)

    print(f"Last rows saved to: {last_rows_csv_path} ({len(last_rows)} rows)")
    print(f"Train data saved to: {train_csv_path} ({len(train_df)} rows)")

def split_and_save(preprocessed_csv_path: str, from_ts: int, to_ts: int):
    """
    Load the preprocessed data, split into train/test based on from/to timestamps,
    and save each as CSV.
    """
    df = load_csv(preprocessed_csv_path)

    # Perform the split
    train_df, test_df = time_based_split(df, from_ts, to_ts)

    save_csv(train_df, str(TRAIN_CSV_PATH))
    save_csv(test_df, str(TEST_CSV_PATH))

    print(f"--- Split Complete ---")
    print(f"Train Window:  Start -> {from_ts}")
    print(f"Test Window:   {from_ts} -> {to_ts}")
    print(f"Dropped Data:  {to_ts} -> End")
    print(f"----------------------")
    print(f"Train CSV saved to: {TRAIN_CSV_PATH} ({len(train_df)} rows)")
    print(f"Test CSV saved to:  {TEST_CSV_PATH} ({len(test_df)} rows)")


if __name__ == "__main__":
    FROM_TS = 1715000000000 - (1000 * 60 * 60 * 24 * 365)  # Start of Test Data, -1 year
    TO_TS = 1715000000000  # End of Test Data

    print("Loading preprocessed data:", PREPROCESSED_CSV_PATH)
    split_and_save(str(PREPROCESSED_CSV_PATH), FROM_TS, TO_TS)