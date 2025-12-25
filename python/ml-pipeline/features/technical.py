"""Technical analysis features.

Includes standard momentum, volatility, and trend indicators.
"""

import pandas as pd
import numpy as np

from config.columns import CLOSE, OPEN, HIGH, LOW, VOLUME, TICKER, TIMESTAMP
from config.settings import EPSILON


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical analysis features to the DataFrame.
    
    Calculates features per ticker to avoid mixing data across assets.
    
    Args:
        df: Wide format DataFrame with OHLCV columns.
    
    Returns:
        DataFrame with technical features added.
    """
    # We need to process by ticker
    if TICKER not in df.columns:
        return df
        
    # Sort by timestamp to ensure correct rolling calculations
    df = df.sort_values([TICKER, TIMESTAMP])
    
    result_dfs = []
    
    for ticker in df[TICKER].unique():
        ticker_df = df[df[TICKER] == ticker].copy()
        
        # Skip if not enough data
        if len(ticker_df) < 50:
            result_dfs.append(ticker_df)
            continue
            
        # --- Momentum ---
        
        # RSI (14 day)
        ticker_df = _add_rsi(ticker_df, window=14)
        
        # MACD (12, 26, 9)
        ticker_df = _add_macd(ticker_df, fast=12, slow=26, signal=9)
        
        # ROC (Rate of Change) - 10 day and 252 day (1 year)
        for window in [10, 252]:
            ticker_df = _add_roc(ticker_df, window=window)
        
        # --- Volatility ---
        
        # ATR (14 day)
        ticker_df = _add_atr(ticker_df, window=14)
        
        # Bollinger Bands Width (20 day)
        ticker_df = _add_bollinger_width(ticker_df, window=20, num_std=2)
        
        # --- Trend / Rolling Stats ---
        
        # Distance from Moving Averages (Short and Long Term)
        for window in [20, 50, 200]:
            ticker_df = _add_ma_distance(ticker_df, window)
            
        # Rolling Volatility (Standard Deviation of returns)
        # 20 day (1 month) and 252 day (1 year)
        for window in [20, 252]:
            ticker_df = _add_rolling_volatility(ticker_df, window=window)
            
        # 52-week High/Low Position
        ticker_df = _add_52week_high_low(ticker_df)
        
        # --- Lagged Returns ---
        
        # Returns for t-1, t-2, t-3, t-5
        for lag in [1, 2, 3, 5]:
            ticker_df = _add_lagged_return(ticker_df, lag)
            
        result_dfs.append(ticker_df)
        
    # Combine back
    if not result_dfs:
        return df
        
    return pd.concat(result_dfs).sort_values(TIMESTAMP).reset_index(drop=True)


def _add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate Relative Strength Index."""
    if CLOSE not in df.columns:
        return df
        
    delta = df[CLOSE].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / (loss + EPSILON)
    df[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    
    # Fill NaN (start of series) with 50 (neutral)
    df[f"RSI_{window}"] = df[f"RSI_{window}"].fillna(50)
    
    return df


def _add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    if CLOSE not in df.columns:
        return df
        
    # Use EMA for MACD
    ema_fast = df[CLOSE].ewm(span=fast, adjust=False).mean()
    ema_slow = df[CLOSE].ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    df["MACD_Line"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Hist"] = macd_line - signal_line
    
    return df


def _add_roc(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Calculate Rate of Change (Percentage change over window)."""
    if CLOSE not in df.columns:
        return df
        
    df[f"ROC_{window}"] = df[CLOSE].pct_change(periods=window) * 100
    return df


def _add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate Average True Range."""
    required = [HIGH, LOW, CLOSE]
    if not all(c in df.columns for c in required):
        return df
        
    high = df[HIGH]
    low = df[LOW]
    close_prev = df[CLOSE].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f"ATR_{window}"] = tr.rolling(window=window).mean()
    
    # Normalize ATR by price to make it comparable across tickers/time
    df[f"NATR_{window}"] = (df[f"ATR_{window}"] / (df[CLOSE] + EPSILON)) * 100
    
    return df


def _add_bollinger_width(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """Calculate Bollinger Band Width (volatility measure)."""
    if CLOSE not in df.columns:
        return df
        
    sma = df[CLOSE].rolling(window=window).mean()
    std = df[CLOSE].rolling(window=window).std()
    
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    
    # Bandwidth: (Upper - Lower) / Middle
    df[f"BB_Width_{window}"] = (upper - lower) / (sma + EPSILON)
    
    # Also add distance from SMA (trend)
    df[f"Dist_SMA_{window}"] = (df[CLOSE] - sma) / (sma + EPSILON)
    
    return df


def _add_ma_distance(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate percentage distance from Moving Average."""
    if CLOSE not in df.columns:
        return df
        
    ma = df[CLOSE].rolling(window=window).mean()
    df[f"Dist_MA_{window}"] = (df[CLOSE] - ma) / (ma + EPSILON)
    
    return df


def _add_rolling_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate rolling standard deviation of returns."""
    if CLOSE not in df.columns:
        return df
        
    returns = df[CLOSE].pct_change()
    df[f"Vol_{window}"] = returns.rolling(window=window).std()
    
    return df


def _add_52week_high_low(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate position relative to 52-week High and Low."""
    if CLOSE not in df.columns:
        return df
        
    window = 252  # Approx 1 trading year
    
    rolling_high = df[HIGH].rolling(window=window).max()
    rolling_low = df[LOW].rolling(window=window).min()
    
    # Position within range (0 = at low, 1 = at high)
    range_size = rolling_high - rolling_low
    df["Pos_52w_Range"] = (df[CLOSE] - rolling_low) / (range_size + EPSILON)
    
    # Distance from High (negative value)
    df["Dist_52w_High"] = (df[CLOSE] - rolling_high) / (rolling_high + EPSILON)
    
    return df


def _add_lagged_return(df: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Calculate lagged returns."""
    if CLOSE not in df.columns:
        return df
        
    # Return from t-lag to t
    # Actually, we want the return that happened AT t-lag
    # So we calculate 1-day return, then shift it
    
    daily_return = df[CLOSE].pct_change()
    df[f"Ret_Lag_{lag}"] = daily_return.shift(lag)
    
    return df
