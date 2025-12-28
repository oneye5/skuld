"""Module for splitting data by timestamp for train/test."""

from dataclasses import dataclass
import pandas as pd

from config.columns import TIMESTAMP
from config.settings import MS_PER_DAY


@dataclass
class TrainTestSplit:
    """Container for train/test split with metadata."""
    train: pd.DataFrame
    test: pd.DataFrame
    train_start_ts: int
    train_end_ts: int
    test_start_ts: int
    test_end_ts: int


def split_by_timestamp(
    df: pd.DataFrame,
    train_end_ts: int,
    test_end_ts: int,
) -> TrainTestSplit:
    """Split data by timestamp into train and test sets.
    
    Train: all data with timestamp < train_end_ts
    Test: data with train_end_ts <= timestamp < test_end_ts
    
    Args:
        df: DataFrame with timestamp column.
        train_end_ts: End timestamp for training data (exclusive).
        test_end_ts: End timestamp for test data (exclusive).
    
    Returns:
        TrainTestSplit with train and test DataFrames.
    """
    train = df[df[TIMESTAMP] < train_end_ts].copy()
    test = df[(df[TIMESTAMP] >= train_end_ts) & (df[TIMESTAMP] < test_end_ts)].copy()
    
    train_start_ts = int(train[TIMESTAMP].min()) if not train.empty else 0
    test_start_ts = int(test[TIMESTAMP].min()) if not test.empty else train_end_ts
    
    return TrainTestSplit(
        train=train,
        test=test,
        train_start_ts=train_start_ts,
        train_end_ts=train_end_ts,
        test_start_ts=test_start_ts,
        test_end_ts=test_end_ts,
    )


def calculate_window_timestamps(
    data_max_ts: int,
    num_windows: int,
    window_movement_years: float,
    lookahead_days: int,
    test_period_years: float,
) -> list[tuple[int, int]]:
    """Calculate train_end and test_end timestamps for rolling windows.
    
    Windows move backward in time from the most recent data.
    Test data must have lookahead_days of future data for labeling.
    
    Args:
        data_max_ts: Maximum timestamp in the dataset.
        num_windows: Number of rolling windows.
        window_movement_years: How far back each window moves.
        lookahead_days: Days needed for lookahead (affects test period end).
        test_period_years: Length of test period in years.
    
    Returns:
        List of (train_end_ts, test_end_ts) tuples.
    """
    window_movement_ms = int(window_movement_years * 365.25 * MS_PER_DAY)
    lookahead_ms = lookahead_days * MS_PER_DAY
    test_period_ms = int(test_period_years * 365.25 * MS_PER_DAY)
    
    # Most recent test_end must leave room for lookahead
    latest_test_end = data_max_ts - lookahead_ms
    
    windows = []
    
    for i in range(num_windows):
        # Test end moves backward for each window
        test_end_ts = latest_test_end - (i * window_movement_ms)
        
        # Train end is test_end minus test_period
        train_end_ts = test_end_ts - test_period_ms
        
        windows.append((train_end_ts, test_end_ts))
    
    return windows
