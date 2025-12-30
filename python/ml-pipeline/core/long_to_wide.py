"""Module for converting long format data to wide format."""

import re
from urllib.parse import unquote

import pandas as pd

from config.columns import TIMESTAMP, TICKER, FEATURE, VALUE, CLOSE, MACRO_PREFIX, OPEN, HIGH, LOW, VOLUME


# OHLCV features that occur together at the same timestamp (no backfill needed)
OHLCV_FEATURES = {CLOSE, OPEN, HIGH, LOW, VOLUME}


def clean_and_classify_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean ticker names and classify non-NZ tickers as macro data.
    
    Converts non-NZX tickers (indexes, forex, commodities, etc.) into macro features
    that will be backfilled to all NZX ticker rows based on timestamp.
    
    Non-NZX ticker patterns detected:
    - Forex: ends with '=X' (e.g., NZDUSD=X)
    - Futures/Commodities: ends with '=F' (e.g., GC=F for gold, CL=F for oil)
    - Global indexes: starts with '%5E' or '^' (e.g., ^TNX, ^FTSE)
    - Shanghai: ends with '.SS' (e.g., 000001.SS)
    
    NZX tickers: end with '.NZ' (e.g., AIR.NZ, FPH.NZ)
    
    Args:
        df: Long format DataFrame with ticker column.
    
    Returns:
        DataFrame with non-NZ tickers converted to macro features.
    """
    df = df.copy()
    
    # Vectorized URL-decode for tickers containing '%' (e.g., %5ETNX -> ^TNX)
    # Only apply unquote to rows that contain '%' to avoid unnecessary processing
    has_encoded = df[TICKER].str.contains('%', na=False)
    if has_encoded.any():
        # Use pandas vectorized string operations where possible
        # For URL decoding, we need to use a map since unquote isn't vectorized
        encoded_tickers = df.loc[has_encoded, TICKER].unique()
        decode_map = {t: unquote(t) for t in encoded_tickers}
        df.loc[has_encoded, TICKER] = df.loc[has_encoded, TICKER].map(decode_map)
    
    # Vectorized identification of NZX tickers (ends with .NZ)
    # Using str.endswith() which is much faster than regex or apply()
    ticker_col = df[TICKER].fillna('')
    is_nzx = ticker_col.str.endswith('.NZ')
    is_empty = ticker_col == ''
    
    # Non-NZX tickers: not .NZ and not empty (these become macro features)
    is_non_nzx = ~is_nzx & ~is_empty
    
    if is_non_nzx.any():
        # For non-NZX tickers, convert to macro features:
        # feature = ticker + "_" + original_feature (e.g., "^TNX_Close")
        # ticker = "" (will get MACRO_ prefix later)
        df.loc[is_non_nzx, FEATURE] = (
            df.loc[is_non_nzx, TICKER] + "_" + df.loc[is_non_nzx, FEATURE]
        )
        df.loc[is_non_nzx, TICKER] = ""
    
    return df


def add_macro_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Add MACRO_ prefix to features with empty ticker.
    
    This distinguishes global/macro features from ticker-specific features
    after pivoting to wide format.
    
    Args:
        df: Long format DataFrame with ticker and feature columns.
    
    Returns:
        DataFrame with prefixed feature names for empty-ticker rows.
    """
    df = df.copy()
    
    # Identify rows with empty ticker (macro/global features)
    is_macro = df[TICKER] == ""
    
    # Add prefix to feature names for macro rows
    df.loc[is_macro, FEATURE] = MACRO_PREFIX + df.loc[is_macro, FEATURE]
    
    return df


def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long format data to wide format with backfilling.
    
    Uses 'Close' observations as timestamp anchors. For each ticker's Close 
    timestamp:
    - OHLCV features use exact timestamp match (they occur together)
    - All other features use most recent value at or before timestamp (backfill)
    
    This matches nzx-predictor's Java TimeSeriesInterpolator.getMostRecent() behavior.
    
    Args:
        df: Long format DataFrame with timestamp, ticker, feature, value columns.
            Should already have MACRO_ prefix added via add_macro_prefix().
    
    Returns:
        Wide format DataFrame with timestamp, ticker, and feature columns.
    """
    # Separate ticker data and macro data
    is_macro = df[FEATURE].str.startswith(MACRO_PREFIX)
    ticker_df = df.loc[~is_macro].copy()
    macro_df = df.loc[is_macro].copy()
    
    # Get Close timestamps as anchors (only for non-empty tickers)
    close_rows = ticker_df.loc[
        (ticker_df[FEATURE] == CLOSE) & (ticker_df[TICKER] != ""),
        [TIMESTAMP, TICKER]
    ].drop_duplicates()
    
    if close_rows.empty:
        return pd.DataFrame(columns=[TIMESTAMP, TICKER])
    
    # Pivot ticker data to wide format WITH backfilling
    ticker_wide = _pivot_ticker_data_with_backfill(ticker_df, close_rows)
    
    # Merge macro data (already uses backfill)
    if not macro_df.empty:
        ticker_wide = _merge_macro_data(ticker_wide, macro_df)
    
    # Convert numeric columns to float32 to save memory
    for col in ticker_wide.columns:
        if col not in [TIMESTAMP, TICKER] and ticker_wide[col].dtype == "float64":
            ticker_wide[col] = ticker_wide[col].astype("float32")
    
    return ticker_wide.reset_index(drop=True)


def _pivot_ticker_data_with_backfill(ticker_df: pd.DataFrame, close_rows: pd.DataFrame) -> pd.DataFrame:
    """Pivot ticker data using Close timestamps as anchors, with backfilling.
    
    OHLCV features use exact timestamp match.
    All other features (fundamentals, etc.) use most recent value at or before timestamp.
    
    Optimized: Uses vectorized forward-fill instead of per-feature merge_asof loops.
    """
    # Filter to only tickers with Close data
    valid_tickers = close_rows[TICKER].unique()
    ticker_df = ticker_df[ticker_df[TICKER].isin(valid_tickers)]
    
    # Separate OHLCV from other features
    is_ohlcv = ticker_df[FEATURE].isin(OHLCV_FEATURES)
    ohlcv_df = ticker_df[is_ohlcv].copy()
    other_df = ticker_df[~is_ohlcv].copy()
    
    # Start with anchor timestamps
    result = close_rows.copy().sort_values([TICKER, TIMESTAMP])
    
    # Pivot OHLCV data (exact timestamp match)
    if not ohlcv_df.empty:
        ohlcv_pivoted = ohlcv_df.pivot_table(
            index=[TICKER, TIMESTAMP],
            columns=FEATURE,
            values=VALUE,
            aggfunc="last",
        ).reset_index()
        result = result.merge(ohlcv_pivoted, on=[TICKER, TIMESTAMP], how="left")
    
    # Backfill other features using vectorized approach
    if not other_df.empty:
        result = _backfill_features_vectorized(result, other_df)
    
    return result


def _backfill_features_vectorized(anchor_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Backfill non-OHLCV features using vectorized forward-fill.
    
    Much faster than per-feature merge_asof loops.
    Strategy:
    1. Pivot all features to wide format
    2. Combine with anchor timestamps
    3. Sort by ticker+timestamp and forward-fill within each ticker
    4. Filter back to just anchor timestamps
    """
    # Pivot all features at once
    features_wide = features_df.pivot_table(
        index=[TICKER, TIMESTAMP],
        columns=FEATURE,
        values=VALUE,
        aggfunc="last",
    ).reset_index()
    
    # Mark anchor rows
    anchor_df = anchor_df.copy()
    anchor_df["_is_anchor"] = True
    
    # Combine anchors with feature data
    combined = pd.concat([
        anchor_df,
        features_wide.assign(_is_anchor=False)
    ], ignore_index=True)
    
    # Sort by ticker and timestamp
    combined = combined.sort_values([TICKER, TIMESTAMP])
    
    # Forward-fill within each ticker (this is the backfill - most recent value)
    feature_cols = [c for c in features_wide.columns if c not in [TICKER, TIMESTAMP]]
    combined[feature_cols] = combined.groupby(TICKER)[feature_cols].ffill()
    
    # Filter back to just anchor rows
    result = combined[combined["_is_anchor"] == True].drop(columns=["_is_anchor"])
    
    return result.reset_index(drop=True)


def _merge_macro_data(ticker_wide: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """Merge macro data using merge_asof (most recent value at or before timestamp).
    
    Uses backward merge_asof first, then forward-fills any remaining NaN.
    This ensures early stock data (before first macro observation) gets filled.
    """
    if ticker_wide.empty or macro_df.empty:
        return ticker_wide
    
    # Pivot macro data to wide format
    macro_pivoted = macro_df.pivot_table(
        index=TIMESTAMP,
        columns=FEATURE,
        values=VALUE,
        aggfunc="last",
    ).reset_index()
    
    macro_cols = [c for c in macro_pivoted.columns if c != TIMESTAMP]
    
    # Sort both for merge_asof
    ticker_wide = ticker_wide.sort_values(TIMESTAMP)
    macro_pivoted = macro_pivoted.sort_values(TIMESTAMP)
    
    # Merge: for each ticker timestamp, get most recent macro data
    result = pd.merge_asof(
        ticker_wide,
        macro_pivoted,
        on=TIMESTAMP,
        direction="backward",
    )
    
    # Forward-fill any remaining NaN in macro columns
    # (for early stock data before first macro observation)
    result[macro_cols] = result[macro_cols].ffill()
    
    # DO NOT backward-fill (bfill) here - that would leak future macro data to the past!
    # If macro data starts later than stock data, early rows should remain NaN
    # (they will be handled by preprocess_data later, likely filled with 0)
    
    return result
