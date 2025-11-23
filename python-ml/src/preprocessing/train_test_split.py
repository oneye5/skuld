from pathlib import Path
import pandas as pd

from src.config.config import *
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


def split_and_save(preprocessed_csv_path: str, from_ts: int, to_ts: int):
    """
    Load the preprocessed data, split into train/test based on from/to timestamps,
    and save each as CSV.
    """
    df = load_csv(preprocessed_csv_path)

    # Perform the split
    train_df, test_df = time_based_split(df, from_ts, to_ts)

    root = get_skuld_root()
    train_csv = root / "python-ml" / "data" / "train.csv"
    test_csv = root / "python-ml" / "data" / "test.csv"

    save_csv(train_df, str(train_csv))
    save_csv(test_df, str(test_csv))

    print(f"--- Split Complete ---")
    print(f"Train Window:  Start -> {from_ts}")
    print(f"Test Window:   {from_ts} -> {to_ts}")
    print(f"Dropped Data:  {to_ts} -> End")
    print(f"----------------------")
    print(f"Train CSV saved to: {train_csv} ({len(train_df)} rows)")
    print(f"Test CSV saved to:  {test_csv} ({len(test_df)} rows)")


if __name__ == "__main__":
    FROM_TS = 1715000000000 - (1000 * 60 * 60 * 24 * 365)  # Start of Test Data, -1 year
    TO_TS = 1715000000000  # End of Test Data

    print("Loading preprocessed data:", PREPROCESSED_CSV_PATH)
    split_and_save(str(PREPROCESSED_CSV_PATH), FROM_TS, TO_TS)