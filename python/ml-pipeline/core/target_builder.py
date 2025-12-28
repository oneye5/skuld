"""Module for computing forward returns for ranking-based prediction.

This module computes continuous forward returns (target for ranking) as opposed to
binary labels used in classification. The forward return represents the price
change over a specified lookahead period.
"""

import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER, CLOSE
from config.settings import MS_PER_DAY


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
) -> pd.DataFrame:
    """Compute forward returns for each ticker.
    
    For each observation, calculates the return over the lookahead period.
    This is the continuous target used for ranking models.
    
    Args:
        df: Wide format DataFrame with timestamp, ticker, and Close columns.
        lookahead_days: Number of days to compute forward return.
        return_type: "simple" for (P_t+n - P_t) / P_t, "log" for ln(P_t+n / P_t).
        winsorize_limits: Optional tuple (lower, upper) to clip extreme returns.
                         E.g., (-0.5, 0.5) clips returns to [-50%, +50%].
        drop_na: If True, drop rows where forward return cannot be computed.
        price_lookup_df: Optional DataFrame with future price data for lookup.
                        If None, uses df for both rows and price lookup.
        tolerance_days: Max days to look past target date for a price.
                       If None, defaults to lookahead_days // 2 + 5.
    
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
            # No lookup data - set all forward returns to NaN
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
        # No data - return empty DataFrame with expected column
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
