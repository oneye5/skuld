"""Cross-sectional features comparing each ticker to the broader market.

Research shows that relative performance (vs market/peers) is often more
predictive than absolute performance. These features capture:
- Rank within the market
- Deviation from market average
- Sector/market regime indicators
"""

import pandas as pd
import numpy as np

from config.column_names import TIMESTAMP, TICKER, CLOSE


# Feature name constants  
RETURN_RANK_5D = "return_rank_5d"
RETURN_RANK_20D = "return_rank_20d"
RETURN_VS_MARKET_5D = "return_vs_market_5d"
RETURN_VS_MARKET_20D = "return_vs_market_20d"
RSI_RANK = "rsi_rank"
VOL_RANK = "volatility_rank"
MARKET_BREADTH = "market_breadth"
MARKET_MOMENTUM = "market_momentum"
MARKET_VOLATILITY = "market_volatility"


def _calculate_returns(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Calculate returns for each ticker."""
    df = df.sort_values([TICKER, TIMESTAMP])
    
    returns = df.groupby(TICKER)[CLOSE].pct_change(periods) * 100
    return returns


def _calculate_cross_sectional_rank(df: pd.DataFrame, value_col: str, output_col: str) -> pd.DataFrame:
    """Calculate percentile rank within each timestamp."""
    df = df.copy()
    
    # Rank within each timestamp (0 to 1, higher = better)
    df[output_col] = df.groupby(TIMESTAMP)[value_col].rank(pct=True)
    
    return df


def _calculate_market_average(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Calculate market average for each timestamp."""
    return df.groupby(TIMESTAMP)[value_col].transform('mean')


def _calculate_market_breadth(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market breadth (% of stocks with positive returns)."""
    df = df.copy()
    
    # 5-day returns
    df = df.sort_values([TICKER, TIMESTAMP])
    df['_return_5d'] = df.groupby(TICKER)[CLOSE].pct_change(5)
    
    # Count positive vs negative per timestamp
    positive_counts = df[df['_return_5d'] > 0].groupby(TIMESTAMP).size()
    total_counts = df.groupby(TIMESTAMP).size()
    
    breadth = positive_counts / total_counts
    breadth = breadth.fillna(0.5)
    
    # Map back to DataFrame
    df[MARKET_BREADTH] = df[TIMESTAMP].map(breadth)
    df = df.drop('_return_5d', axis=1)
    
    return df


def _calculate_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market regime indicators."""
    df = df.copy()
    
    # Calculate equal-weighted market return and volatility
    df = df.sort_values([TICKER, TIMESTAMP])
    df['_return_1d'] = df.groupby(TICKER)[CLOSE].pct_change(1)
    
    # Market momentum: average 20-day return across all tickers
    df['_return_20d'] = df.groupby(TICKER)[CLOSE].pct_change(20)
    market_mom = df.groupby(TIMESTAMP)['_return_20d'].transform('mean') * 100
    df[MARKET_MOMENTUM] = market_mom
    
    # Market volatility: cross-sectional dispersion of returns
    market_vol = df.groupby(TIMESTAMP)['_return_1d'].transform('std') * 100
    df[MARKET_VOLATILITY] = market_vol
    
    df = df.drop(['_return_1d', '_return_20d'], axis=1)
    
    return df


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-sectional features comparing each ticker to the market.
    
    Args:
        df: Wide format DataFrame with at least CLOSE column and multiple tickers.
    
    Returns:
        DataFrame with additional cross-sectional features.
    """
    if df.empty or CLOSE not in df.columns:
        return df
    
    # Need multiple tickers for cross-sectional analysis
    unique_tickers = df[TICKER].nunique()
    if unique_tickers < 5:
        return df
    
    df = df.sort_values([TIMESTAMP, TICKER]).copy()
    
    # Calculate returns first (needed for other features)
    df['_return_5d'] = df.groupby(TICKER)[CLOSE].pct_change(5) * 100
    df['_return_20d'] = df.groupby(TICKER)[CLOSE].pct_change(20) * 100
    
    # Cross-sectional ranks - vectorized operations
    df[RETURN_RANK_5D] = df.groupby(TIMESTAMP)['_return_5d'].rank(pct=True)
    df[RETURN_RANK_20D] = df.groupby(TIMESTAMP)['_return_20d'].rank(pct=True)
    
    # Return vs market average
    market_avg_5d = df.groupby(TIMESTAMP)['_return_5d'].transform('mean')
    market_avg_20d = df.groupby(TIMESTAMP)['_return_20d'].transform('mean')
    df[RETURN_VS_MARKET_5D] = df['_return_5d'] - market_avg_5d
    df[RETURN_VS_MARKET_20D] = df['_return_20d'] - market_avg_20d
    
    # RSI rank if available
    if 'rsi_14' in df.columns:
        df[RSI_RANK] = df.groupby(TIMESTAMP)['rsi_14'].rank(pct=True)
    
    # Volatility rank if available
    if 'volatility_20d' in df.columns:
        df[VOL_RANK] = df.groupby(TIMESTAMP)['volatility_20d'].rank(pct=True)
    
    # Market breadth - vectorized approach
    df['_positive'] = (df['_return_5d'] > 0).astype(float)
    df[MARKET_BREADTH] = df.groupby(TIMESTAMP)['_positive'].transform('mean')
    df = df.drop('_positive', axis=1)
    
    # Market regime features
    df[MARKET_MOMENTUM] = df.groupby(TIMESTAMP)['_return_20d'].transform('mean')
    daily_ret = df.groupby(TICKER)[CLOSE].pct_change(1)
    df['_daily_ret'] = daily_ret
    df[MARKET_VOLATILITY] = df.groupby(TIMESTAMP)['_daily_ret'].transform('std') * 100
    
    # Clean up temp columns
    df = df.drop(['_return_5d', '_return_20d', '_daily_ret'], axis=1)
    
    # Convert to float32 and handle infinities
    float_cols = [RETURN_RANK_5D, RETURN_RANK_20D, RETURN_VS_MARKET_5D, 
                  RETURN_VS_MARKET_20D, RSI_RANK, VOL_RANK,
                  MARKET_BREADTH, MARKET_MOMENTUM, MARKET_VOLATILITY]
    
    for col in float_cols:
        if col in df.columns:
            # Replace infinities with NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].astype('float32')
    
    return df


def get_cross_sectional_feature_columns() -> list[str]:
    """Return list of cross-sectional feature column names."""
    return [
        RETURN_RANK_5D, RETURN_RANK_20D,
        RETURN_VS_MARKET_5D, RETURN_VS_MARKET_20D,
        RSI_RANK, VOL_RANK,
        MARKET_BREADTH, MARKET_MOMENTUM, MARKET_VOLATILITY,
    ]
