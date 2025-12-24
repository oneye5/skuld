"""Time-based features."""

import pandas as pd

from config.columns import TIMESTAMP, TIME_SCALED


def add_time_features(
    df: pd.DataFrame,
    time_min: int | None = None,
    time_max: int | None = None,
) -> pd.DataFrame:
    """Add time-based features.
    
    Adds a min-max scaled time feature to preserve temporal information
    while keeping values in a normalized range.
    
    Args:
        df: DataFrame with timestamp column.
        time_min: Minimum timestamp for scaling. If None, uses df min.
        time_max: Maximum timestamp for scaling. If None, uses df max.
    
    Returns:
        DataFrame with time features added.
    """
    result = df.copy()
    
    # Use provided bounds or calculate from data
    t_min = time_min if time_min is not None else result[TIMESTAMP].min()
    t_max = time_max if time_max is not None else result[TIMESTAMP].max()
    
    # Avoid division by zero
    if t_max == t_min:
        result[TIME_SCALED] = 0.5
    else:
        result[TIME_SCALED] = (result[TIMESTAMP] - t_min) / (t_max - t_min)
    
    return result


def get_time_bounds(df: pd.DataFrame) -> tuple[int, int]:
    """Get min and max timestamps from DataFrame.
    
    Args:
        df: DataFrame with timestamp column.
    
    Returns:
        Tuple of (min_timestamp, max_timestamp).
    """
    return int(df[TIMESTAMP].min()), int(df[TIMESTAMP].max())
