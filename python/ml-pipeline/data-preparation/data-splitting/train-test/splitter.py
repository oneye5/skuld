"""Module for time-based train/test splitting."""

from dataclasses import dataclass
import pandas as pd

from config.column_names import TIMESTAMP


@dataclass
class TrainTestSplit:
    """Container for train and test data splits."""
    train: pd.DataFrame
    test: pd.DataFrame
    train_start_ts: int
    train_end_ts: int
    test_start_ts: int
    test_end_ts: int


def split_by_timestamp(
    df: pd.DataFrame,
    train_end_ts: int,
    test_start_ts: int | None = None,
    test_end_ts: int | None = None,
) -> TrainTestSplit:
    """
    Split data by timestamp into train and test sets.
    
    Args:
        df: Wide format DataFrame with timestamp column.
        train_end_ts: Timestamp (exclusive) marking end of training period.
        test_start_ts: Timestamp (inclusive) marking start of test period.
                      Defaults to train_end_ts if not provided.
        test_end_ts: Timestamp (exclusive) marking end of test period.
                    Uses all remaining data if not provided.
    
    Returns:
        TrainTestSplit containing train and test DataFrames.
    """
    if test_start_ts is None:
        test_start_ts = train_end_ts
    
    train_mask = df[TIMESTAMP] < train_end_ts
    train_df = df[train_mask].copy()
    
    test_mask = df[TIMESTAMP] >= test_start_ts
    if test_end_ts is not None:
        test_mask = test_mask & (df[TIMESTAMP] < test_end_ts)
    test_df = df[test_mask].copy()
    
    train_start_ts = int(train_df[TIMESTAMP].min()) if len(train_df) > 0 else 0
    actual_train_end = int(train_df[TIMESTAMP].max()) if len(train_df) > 0 else train_end_ts
    actual_test_start = int(test_df[TIMESTAMP].min()) if len(test_df) > 0 else test_start_ts
    actual_test_end = int(test_df[TIMESTAMP].max()) if len(test_df) > 0 else (test_end_ts or 0)
    
    return TrainTestSplit(
        train=train_df,
        test=test_df,
        train_start_ts=train_start_ts,
        train_end_ts=actual_train_end,
        test_start_ts=actual_test_start,
        test_end_ts=actual_test_end,
    )
