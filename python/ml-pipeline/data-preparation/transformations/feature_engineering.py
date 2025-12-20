"""Module for feature engineering - adding derived features."""

import pandas as pd
import numpy as np

from config.column_names import (
    TIMESTAMP,
    DAY_OF_YEAR_SIN,
    DAY_OF_YEAR_COS,
    DAY_OF_WEEK_SIN,
    DAY_OF_WEEK_COS,
    MONTH_SIN,
    MONTH_COS,
)


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical time-based features derived from timestamp.

    Cyclical encoding using sin/cos ensures continuity (e.g., Dec 31 is 
    close to Jan 1).

    Args:
        df: DataFrame with timestamp column (Unix timestamp in milliseconds).

    Returns:
        DataFrame with added cyclical time features.
    """
    # Convert timestamp to datetime
    dt = pd.to_datetime(df[TIMESTAMP], unit='ms')

    # Day of year (1-366)
    day_of_year = dt.dt.dayofyear
    # Day of week (0-6)
    day_of_week = dt.dt.dayofweek
    # Month (1-12)
    month = dt.dt.month

    # Create all features at once to avoid fragmentation
    new_features = pd.DataFrame({
        DAY_OF_YEAR_SIN: np.sin(2 * np.pi * day_of_year / 365.25),
        DAY_OF_YEAR_COS: np.cos(2 * np.pi * day_of_year / 365.25),
        DAY_OF_WEEK_SIN: np.sin(2 * np.pi * day_of_week / 7),
        DAY_OF_WEEK_COS: np.cos(2 * np.pi * day_of_week / 7),
        MONTH_SIN: np.sin(2 * np.pi * month / 12),
        MONTH_COS: np.cos(2 * np.pi * month / 12),
    }, index=df.index)

    return pd.concat([df, new_features], axis=1)
