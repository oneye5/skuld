"""Module for converting long format data to wide format."""

import pandas as pd

from config.columns import TIMESTAMP, TICKER, FEATURE, VALUE, CLOSE, MACRO_PREFIX, OPEN, HIGH, LOW, VOLUME


# OHLCV features that occur together at the same timestamp (no backfill needed)
OHLCV_FEATURES = {CLOSE, OPEN, HIGH, LOW, VOLUME}


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
    """Merge macro data using merge_asof (most recent value at or before timestamp)."""
    if ticker_wide.empty or macro_df.empty:
        return ticker_wide
    
    # Pivot macro data to wide format
    macro_pivoted = macro_df.pivot_table(
        index=TIMESTAMP,
        columns=FEATURE,
        values=VALUE,
        aggfunc="last",
    ).reset_index()
    
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
    
    return result
