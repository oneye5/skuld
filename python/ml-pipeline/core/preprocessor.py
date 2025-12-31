"""Module for data preprocessing (cleaning, NaN handling, anomaly detection)."""

import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Optional

from config.columns import TIMESTAMP, TICKER, TARGET
from config.settings import CLIP_THRESHOLD


# =============================================================================
# ANOMALY DETECTION FOR DATA QUALITY
# =============================================================================

def detect_price_anomalies(
    df: pd.DataFrame,
    price_col: str = "Close",
    return_threshold: float = 2.0,
    timestamp_col: str = TIMESTAMP,
    ticker_col: str = TICKER,
) -> pd.DataFrame:
    """Detect anomalous price moves indicating data quality issues.
    
    Identifies rows where daily price returns exceed a threshold, suggesting
    unadjusted stock splits, ticker recycling, or data errors.
    
    Args:
        df: Wide-format DataFrame with price data.
        price_col: Column name for price (default 'Close').
        return_threshold: Absolute return threshold to flag (default 2.0 = 200%).
        timestamp_col: Column name for timestamp.
        ticker_col: Column name for ticker.
    
    Returns:
        DataFrame with additional columns:
            - _daily_return: Computed daily return
            - _is_anomaly: Boolean flag (True = anomalous data point)
            - _anomaly_timestamp: Timestamp of first anomaly for this ticker (if any)
    """
    if df.empty or price_col not in df.columns:
        return df
    
    result = df.copy()
    
    # Sort by ticker and timestamp
    result = result.sort_values([ticker_col, timestamp_col]).reset_index(drop=True)
    
    # Compute daily returns
    result['_prev_price'] = result.groupby(ticker_col)[price_col].shift(1)
    result['_daily_return'] = (result[price_col] - result['_prev_price']) / result['_prev_price']
    
    # Flag anomalies (extreme returns)
    result['_is_anomaly'] = abs(result['_daily_return']) > return_threshold
    
    # For each ticker, find the FIRST anomaly timestamp
    # This marks the discontinuity point where we should trim
    anomaly_rows = result[result['_is_anomaly']]
    first_anomaly_per_ticker = anomaly_rows.groupby(ticker_col)[timestamp_col].min()
    result['_anomaly_timestamp'] = result[ticker_col].map(first_anomaly_per_ticker)
    
    # Clean up temp column
    result = result.drop(columns=['_prev_price'])
    
    return result


def filter_anomalous_data(
    df: pd.DataFrame,
    trim_before_anomaly: bool = True,
    timestamp_col: str = TIMESTAMP,
    ticker_col: str = TICKER,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter anomalous data by trimming pre-anomaly history.
    
    When a ticker has an anomaly (e.g., ticker recycling, unadjusted split),
    the data before and after represent different price series that shouldn't
    be connected. This function keeps only the NEWER portion (after the anomaly).
    
    Args:
        df: DataFrame with anomaly detection columns (from detect_price_anomalies).
        trim_before_anomaly: If True, remove all data BEFORE the first anomaly
            for each affected ticker (keeps the newer, post-anomaly series).
            If False, just remove the anomalous rows themselves.
        timestamp_col: Column name for timestamp.
        ticker_col: Column name for ticker.
    
    Returns:
        Tuple of (filtered_df, removed_df):
            - filtered_df: Data with old series trimmed
            - removed_df: The removed rows (for inspection)
    """
    if '_is_anomaly' not in df.columns:
        warnings.warn("No _is_anomaly column found. Run detect_price_anomalies first.")
        return df, pd.DataFrame()
    
    if trim_before_anomaly and '_anomaly_timestamp' in df.columns:
        # For tickers with anomalies, keep only data >= anomaly timestamp
        # (i.e., the newer series after the discontinuity)
        has_anomaly = df['_anomaly_timestamp'].notna()
        
        # Keep: no anomaly OR timestamp >= first anomaly timestamp
        keep_mask = ~has_anomaly | (df[timestamp_col] >= df['_anomaly_timestamp'])
        
        filtered = df[keep_mask].copy()
        removed = df[~keep_mask].copy()
    else:
        # Simple approach: just remove anomaly rows
        filtered = df[~df['_is_anomaly']].copy()
        removed = df[df['_is_anomaly']].copy()
    
    # Clean up anomaly metadata columns
    anomaly_cols = ['_daily_return', '_is_anomaly', '_anomaly_timestamp']
    filtered = filtered.drop(columns=[c for c in anomaly_cols if c in filtered.columns])
    
    return filtered, removed


def get_anomaly_summary(
    df: pd.DataFrame,
    ticker_col: str = TICKER,
    timestamp_col: str = TIMESTAMP,
) -> dict:
    """Get summary of anomalies detected in data.
    
    Args:
        df: DataFrame with _is_anomaly column.
        ticker_col: Column name for ticker.
        timestamp_col: Column name for timestamp.
    
    Returns:
        Dictionary with anomaly statistics.
    """
    if '_is_anomaly' not in df.columns:
        return {'error': 'No _is_anomaly column found'}
    
    anomalies = df[df['_is_anomaly']]
    
    # Get tickers that have any anomaly
    affected_tickers = []
    if '_anomaly_timestamp' in df.columns:
        affected_tickers = df[df['_anomaly_timestamp'].notna()][ticker_col].unique().tolist()
    elif ticker_col in anomalies.columns:
        affected_tickers = anomalies[ticker_col].unique().tolist()
    
    # Count rows that would be trimmed (before first anomaly for affected tickers)
    rows_to_trim = 0
    if '_anomaly_timestamp' in df.columns:
        has_anomaly = df['_anomaly_timestamp'].notna()
        before_anomaly = df[timestamp_col] < df['_anomaly_timestamp']
        rows_to_trim = (has_anomaly & before_anomaly).sum()
    
    summary = {
        'total_rows': len(df),
        'anomaly_rows': len(anomalies),
        'rows_to_trim': int(rows_to_trim),
        'trim_pct': rows_to_trim / len(df) * 100 if len(df) > 0 else 0,
        'affected_tickers': affected_tickers,
        'n_affected_tickers': len(affected_tickers),
    }
    
    # Add sample of extreme returns
    if '_daily_return' in anomalies.columns and len(anomalies) > 0:
        extreme_returns = anomalies['_daily_return'].dropna()
        if len(extreme_returns) > 0:
            summary['max_return'] = float(extreme_returns.max())
            summary['min_return'] = float(extreme_returns.min())
    
    return summary


def preprocess_data(df: pd.DataFrame, add_missing_flags: bool = True) -> pd.DataFrame:
    """Preprocess data: handle NaN, infinities, and type conversions.
    
    Following nzx-predictor Java approach exactly:
    - Replace infinities with NaN
    - Add MissingFlag columns (1=present, 0=missing) for ALL numeric features
    - Fill NaN with 0.0
    - Convert booleans to int
    
    The missing flag pattern matches CsvWriter.java:
    - Flag = 1 means data was present (observed)
    - Flag = 0 means data was missing (imputed)
    
    IMPORTANT: Forward fill is applied per-ticker in TIMESTAMP order to prevent
    future data from leaking into past observations.
    
    Args:
        df: DataFrame to preprocess.
        add_missing_flags: If True, add binary flag columns for ALL features.
            This matches nzx-predictor's approach exactly.
    
    Returns:
        Preprocessed DataFrame.
    
    Raises:
        ValueError: If DataFrame is empty or missing required columns.
    """
    if df.empty:
        warnings.warn("preprocess_data received empty DataFrame", UserWarning)
        return df
    
    result = df.copy()
    
    # Remove any unnamed/index columns
    result = result.loc[:, ~result.columns.str.contains("^Unnamed")]
    
    # Convert booleans to integers
    bool_cols = result.select_dtypes(include="bool").columns
    result[bool_cols] = result[bool_cols].astype(int)
    
    # Get numeric columns (excluding metadata)
    excluded = [TIMESTAMP, TICKER, TARGET]
    numeric_cols = [
        c for c in result.columns 
        if c not in excluded and pd.api.types.is_numeric_dtype(result[c])
    ]
    
    # Replace infinities with NaN first
    for col in numeric_cols:
        result[col] = result[col].replace([np.inf, -np.inf], np.nan)
    
    # CRITICAL: Create missing flags BEFORE forward fill
    # This ensures flags reflect the ORIGINAL data availability, not post-imputation state
    # Flag = 1 means data was present (observed)
    # Flag = 0 means data was missing (will be imputed)
    missing_flag_cols = {}
    if add_missing_flags:
        for col in numeric_cols:
            # Create flag: 1 if NOT NaN (present), 0 if NaN (missing)
            present_mask = result[col].notna()
            missing_flag_cols[f"MissingFlag_{col}"] = present_mask.astype(np.uint8)
    
    # Forward fill missing values within each ticker group
    # CRITICAL: Sort by TICKER then TIMESTAMP to ensure forward fill only
    # propagates PAST values to PRESENT (not future to past - would cause leakage)
    if TICKER in result.columns and TIMESTAMP in result.columns:
        # Sort to ensure correct temporal ordering within each ticker
        result = result.sort_values([TICKER, TIMESTAMP]).reset_index(drop=True)
        
        # Forward fill numeric columns by ticker
        # This propagates the last known value (e.g. yesterday's price/rate)
        result[numeric_cols] = result.groupby(TICKER)[numeric_cols].ffill()
    elif TIMESTAMP in result.columns:
        # No ticker column - sort by timestamp only
        result = result.sort_values(TIMESTAMP).reset_index(drop=True)
        result[numeric_cols] = result[numeric_cols].ffill()

    # Add missing flags AFTER forward fill (flags were created before, so they're correct)
    if add_missing_flags and missing_flag_cols:
        # Re-index flags to match the sorted result (important after sorting)
        flag_df = pd.DataFrame(missing_flag_cols, index=result.index)
        result = pd.concat([result, flag_df], axis=1)
    
    # Fill remaining NaN with 0.0 (for values that couldn't be forward filled)
    for col in numeric_cols:
        result[col] = result[col].fillna(0.0)
    
    return result


def clip_extreme_values(
    df: pd.DataFrame,
    threshold: float = CLIP_THRESHOLD,
) -> pd.DataFrame:
    """Clip extreme values in scaled features.
    
    After RobustScaler, most values should be in a reasonable range,
    but extreme outliers can still exist. This clips them to avoid
    model instability.
    
    Args:
        df: DataFrame with scaled features.
        threshold: Values beyond [-threshold, threshold] are clipped.
    
    Returns:
        DataFrame with clipped values.
    """
    result = df.copy()
    
    # Columns to exclude from clipping
    excluded = [TIMESTAMP, TICKER, TARGET]
    
    for col in result.columns:
        if col in excluded:
            continue
        if "MissingFlag" in col:
            continue  # Binary flags, don't clip
        if not pd.api.types.is_numeric_dtype(result[col]):
            continue
        
        # Clip to [-threshold, threshold]
        result[col] = result[col].clip(-threshold, threshold)
    
    return result


def drop_sparse_columns(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Drop columns with too many missing values.
    
    Args:
        df: DataFrame to filter.
        threshold: Maximum fraction of missing values allowed.
    
    Returns:
        DataFrame with sparse columns removed.
    """
    missing_frac = df.isnull().mean()
    cols_to_keep = missing_frac[missing_frac < threshold].index.tolist()
    
    return df[cols_to_keep]
