"""Module for computing forward returns for ranking-based prediction.

This module computes continuous forward returns (target for ranking) as opposed to
binary labels used in classification. The forward return represents the price
change over a specified lookahead period.

Performance: Uses vectorized operations with merge_asof for ~3-5x speedup.
"""

import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER, CLOSE, ADJCLOSE
from config.settings import MS_PER_DAY, RETURN_PRICE_COLUMN


# =============================================================================
# COLUMN NAMES
# =============================================================================
FORWARD_RETURN: str = "forward_return"
"""Column name for the computed forward return."""


def compute_forward_returns(
    df: pd.DataFrame,
    lookahead_days: int = 5,
    return_type: str = "simple",
    winsorize_limits: tuple[float, float] | None = None,
    drop_na: bool = False,
    price_lookup_df: pd.DataFrame | None = None,
    tolerance_days: int | None = None,
    price_column: str | None = None,
) -> pd.DataFrame:
    """Compute forward returns for each ticker using vectorized operations.
    
    For each observation, calculates the return over the lookahead period.
    This is the continuous target used for ranking models.
    
    Args:
        df: Wide format DataFrame with timestamp, ticker, and price columns.
        lookahead_days: Number of days to compute forward return.
        return_type: "simple" for (P_t+n - P_t) / P_t, "log" for ln(P_t+n / P_t).
        winsorize_limits: Optional tuple (lower, upper) to clip extreme returns.
                         E.g., (-0.5, 0.5) clips returns to [-50%, +50%].
        drop_na: If True, drop rows where forward return cannot be computed.
        price_lookup_df: Optional DataFrame with future price data for lookup.
                        If None, uses df for both rows and price lookup.
        tolerance_days: Max days to look past target date for a price.
                       If None, defaults to lookahead_days // 2 + 5.
        price_column: Column to use for price. Defaults to RETURN_PRICE_COLUMN setting.
                     Use 'AdjClose' for total return (includes dividends).
                     Use 'Close' for price-only return.
    
    Returns:
        DataFrame with 'forward_return' column added.
        Rows without valid future data will have NaN in forward_return
        (or be dropped if drop_na=True).
    
    Example:
        >>> df = pd.DataFrame({
        ...     "timestamp": [0, 5 * MS_PER_DAY],
        ...     "ticker": ["A", "A"],
        ...     "Close": [100.0, 110.0],
        ... })
        >>> result = compute_forward_returns(df, lookahead_days=5)
        >>> result["forward_return"].iloc[0]
        0.10  # (110 - 100) / 100
    """
    if return_type not in ("simple", "log"):
        raise ValueError(f"return_type must be 'simple' or 'log', got '{return_type}'")
    
    # Determine which price column to use
    if price_column is None:
        price_column = RETURN_PRICE_COLUMN
    
    # Fallback to Close if AdjClose not available
    if price_column == ADJCLOSE and ADJCLOSE not in df.columns:
        import warnings
        warnings.warn(
            f"'{ADJCLOSE}' column not found, falling back to '{CLOSE}'. "
            "Returns will not account for dividends/splits.",
            UserWarning
        )
        price_column = CLOSE
    
    lookahead_ms = lookahead_days * MS_PER_DAY
    
    # Set default tolerance
    if tolerance_days is None:
        tolerance_days = lookahead_days // 2 + 5
    tolerance_ms = int(tolerance_days * MS_PER_DAY)
    
    # Use provided price_lookup_df or fall back to df
    lookup_df = price_lookup_df if price_lookup_df is not None else df
    
    # Use vectorized implementation
    return _compute_forward_returns_vectorized(
        df=df,
        lookup_df=lookup_df,
        lookahead_ms=lookahead_ms,
        tolerance_ms=tolerance_ms,
        return_type=return_type,
        winsorize_limits=winsorize_limits,
        drop_na=drop_na,
        price_column=price_column,
    )


def _compute_forward_returns_vectorized(
    df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    lookahead_ms: int,
    tolerance_ms: int,
    return_type: str,
    winsorize_limits: tuple[float, float] | None,
    drop_na: bool,
    price_column: str = CLOSE,
) -> pd.DataFrame:
    """Vectorized implementation of forward return computation.
    
    Uses merge_asof per ticker group for efficient lookups.
    Memory optimized: minimizes copies within the loop.
    
    Args:
        price_column: Column to use for price data (e.g., 'Close' or 'AdjClose').
    """
    if price_column not in df.columns or price_column not in lookup_df.columns:
        result = df.copy()
        result[FORWARD_RETURN] = np.nan
        return result
    
    # Prepare the main dataframe - compute target timestamp
    result = df.copy()
    result["_target_ts"] = result[TIMESTAMP] + lookahead_ms
    
    # Prepare lookup dataframe - only keep columns we need (memory optimization)
    lookup_for_merge = lookup_df[[TICKER, TIMESTAMP, price_column]].copy()
    lookup_for_merge = lookup_for_merge.rename(columns={price_column: "_future_price"})
    
    # Process each ticker group using vectorized merge_asof
    result_dfs = []
    tickers = result[TICKER].unique()
    
    for ticker in tickers:
        ticker_mask = result[TICKER] == ticker
        ticker_result = result.loc[ticker_mask]  # View first
        
        lookup_mask = lookup_for_merge[TICKER] == ticker
        ticker_lookup = lookup_for_merge.loc[lookup_mask]
        
        if ticker_lookup.empty:
            # Just add NaN column without copy
            ticker_out = ticker_result.copy()
            ticker_out[FORWARD_RETURN] = np.nan
            result_dfs.append(ticker_out.drop(columns=["_target_ts"]))
            continue
        
        # Sort for merge_asof (need copies here for sorting)
        ticker_result_sorted = ticker_result.sort_values("_target_ts")
        ticker_lookup_sorted = ticker_lookup.sort_values(TIMESTAMP)
        
        # merge_asof finds future price
        merged = pd.merge_asof(
            ticker_result_sorted,
            ticker_lookup_sorted[[TIMESTAMP, "_future_price"]],
            left_on="_target_ts",
            right_on=TIMESTAMP,
            direction="forward",
            tolerance=tolerance_ms,
            suffixes=("", "_lookup"),
        )
        
        # Calculate return using the specified price column
        current_price = merged[price_column].values
        future_price = merged["_future_price"].values
        
        if return_type == "simple":
            forward_return = (future_price - current_price) / current_price
        else:  # log
            with np.errstate(divide='ignore', invalid='ignore'):
                forward_return = np.log(future_price / current_price)
        
        merged[FORWARD_RETURN] = forward_return
        
        # Clean up temp columns
        cols_to_drop = ["_target_ts", "_future_price"]
        cols_to_drop.extend([c for c in merged.columns if c.endswith("_lookup")])
        merged = merged.drop(columns=[c for c in cols_to_drop if c in merged.columns])
        
        result_dfs.append(merged)
    
    # Free intermediate memory
    del lookup_for_merge
    
    if not result_dfs:
        result = df.copy()
        result[FORWARD_RETURN] = np.nan
        return result
    
    # Combine all tickers
    result = pd.concat(result_dfs, ignore_index=True)
    del result_dfs  # Free list memory
    
    # Apply winsorization if specified
    if winsorize_limits is not None:
        lower, upper = winsorize_limits
        result[FORWARD_RETURN] = result[FORWARD_RETURN].clip(lower=lower, upper=upper)
    
    # Drop NaN if requested
    if drop_na:
        result = result.dropna(subset=[FORWARD_RETURN])
    
    return result


def compute_forward_returns_loop(
    df: pd.DataFrame,
    lookahead_days: int = 5,
    return_type: str = "simple",
    winsorize_limits: tuple[float, float] | None = None,
    drop_na: bool = False,
    price_lookup_df: pd.DataFrame | None = None,
    tolerance_days: int | None = None,
) -> pd.DataFrame:
    """Original loop-based implementation (kept for testing/reference).
    
    Loops over tickers individually. Slower but matches original behavior exactly.
    """
    if return_type not in ("simple", "log"):
        raise ValueError(f"return_type must be 'simple' or 'log', got '{return_type}'")
    
    lookahead_ms = lookahead_days * MS_PER_DAY
    
    # Set default tolerance
    if tolerance_days is None:
        tolerance_days = lookahead_days // 2 + 5
    tolerance_ms = int(tolerance_days * MS_PER_DAY)
    
    # Use provided price_lookup_df or fall back to df
    lookup_df = price_lookup_df if price_lookup_df is not None else df
    
    result_dfs = []
    
    for ticker in df[TICKER].unique():
        ticker_df = df[df[TICKER] == ticker].copy()
        ticker_df = ticker_df.sort_values(TIMESTAMP).reset_index(drop=True)
        
        if CLOSE not in ticker_df.columns:
            continue
        
        # Get price lookup data for this ticker
        ticker_lookup = lookup_df[lookup_df[TICKER] == ticker].sort_values(TIMESTAMP)
        
        if CLOSE not in ticker_lookup.columns or len(ticker_lookup) == 0:
            ticker_df[FORWARD_RETURN] = np.nan
            result_dfs.append(ticker_df)
            continue
        
        # Calculate target timestamp for each row
        target_ts = ticker_df[TIMESTAMP] + lookahead_ms
        
        # Create lookup DataFrame
        lookup_for_merge = ticker_lookup[[TIMESTAMP, CLOSE]].copy()
        lookup_for_merge = lookup_for_merge.sort_values(TIMESTAMP)
        
        target_df = pd.DataFrame({
            "target_ts": target_ts.values,
            "orig_idx": ticker_df.index.values,
        }).sort_values("target_ts")
        
        # merge_asof finds the first timestamp >= target_ts (direction='forward')
        merged = pd.merge_asof(
            target_df,
            lookup_for_merge,
            left_on="target_ts",
            right_on=TIMESTAMP,
            direction="forward",
            tolerance=tolerance_ms,
        )
        
        # Reorder to match original index
        merged = merged.set_index("orig_idx").reindex(ticker_df.index)
        future_price = merged[CLOSE].values
        current_price = ticker_df[CLOSE].values
        
        # Calculate return
        if return_type == "simple":
            forward_return = (future_price - current_price) / current_price
        else:  # log
            forward_return = np.log(future_price / current_price)
        
        ticker_df[FORWARD_RETURN] = forward_return
        result_dfs.append(ticker_df)
    
    if not result_dfs:
        result = df.copy()
        result[FORWARD_RETURN] = np.nan
        return result
    
    result = pd.concat(result_dfs, ignore_index=True)
    
    # Apply winsorization if specified
    if winsorize_limits is not None:
        lower, upper = winsorize_limits
        result[FORWARD_RETURN] = result[FORWARD_RETURN].clip(lower=lower, upper=upper)
    
    # Drop NaN if requested
    if drop_na:
        result = result.dropna(subset=[FORWARD_RETURN])
    
    return result


def get_max_forward_timestamp(
    max_timestamp: int,
    lookahead_days: int,
) -> int:
    """Get the maximum timestamp that can have a valid forward return.
    
    Args:
        max_timestamp: Maximum timestamp in the dataset (milliseconds).
        lookahead_days: Number of days for forward return calculation.
    
    Returns:
        Maximum timestamp (in ms) that can be labeled with forward return.
    """
    lookahead_ms = lookahead_days * MS_PER_DAY
    return max_timestamp - lookahead_ms


def compute_cross_sectional_ranks(
    df: pd.DataFrame,
    score_col: str,
    timestamp_col: str = TIMESTAMP,
    ascending: bool = False,
) -> pd.Series:
    """Compute cross-sectional ranks within each timestamp.
    
    Args:
        df: DataFrame with timestamp and score columns.
        score_col: Column name containing scores to rank.
        timestamp_col: Column name for timestamp grouping.
        ascending: If True, lower scores get lower ranks. 
                  If False (default), higher scores get rank 1.
    
    Returns:
        Series with ranks (1 = best) for each row.
    """
    return df.groupby(timestamp_col)[score_col].rank(
        method="average", 
        ascending=ascending
    )
