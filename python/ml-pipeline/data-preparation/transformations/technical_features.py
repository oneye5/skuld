"""Module for technical/price-based feature engineering."""

import pandas as pd
import numpy as np

from config.column_names import TIMESTAMP, TICKER, CLOSE, OPEN, HIGH, LOW, VOLUME


# Feature name constants
RETURN_1D = "return_1d"
RETURN_5D = "return_5d"
RETURN_20D = "return_20d"
RETURN_60D = "return_60d"
RETURN_252D = "return_252d"

VOLATILITY_5D = "volatility_5d"
VOLATILITY_20D = "volatility_20d"
VOLATILITY_60D = "volatility_60d"

SMA_5 = "sma_5"
SMA_20 = "sma_20"
SMA_60 = "sma_60"
SMA_200 = "sma_200"

PRICE_TO_SMA_20 = "price_to_sma_20"
PRICE_TO_SMA_60 = "price_to_sma_60"
PRICE_TO_SMA_200 = "price_to_sma_200"

SMA_20_TO_SMA_60 = "sma_20_to_sma_60"
SMA_60_TO_SMA_200 = "sma_60_to_sma_200"

RSI_14 = "rsi_14"
HIGH_LOW_RANGE = "high_low_range"
CLOSE_TO_HIGH = "close_to_high"
CLOSE_TO_LOW = "close_to_low"

VOLUME_SMA_20 = "volume_sma_20"
VOLUME_RATIO = "volume_ratio"

MOMENTUM_5D = "momentum_5d"
MOMENTUM_20D = "momentum_20d"

# Days since high/low
DAYS_SINCE_52W_HIGH = "days_since_52w_high"
DAYS_SINCE_52W_LOW = "days_since_52w_low"
PCT_FROM_52W_HIGH = "pct_from_52w_high"
PCT_FROM_52W_LOW = "pct_from_52w_low"


def _calculate_returns(series: pd.Series, periods: int) -> pd.Series:
    """Calculate percentage returns over N periods."""
    return series.pct_change(periods) * 100


def _calculate_volatility(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling volatility (std of returns)."""
    returns = series.pct_change()
    return returns.rolling(window=window, min_periods=max(1, window // 2)).std() * 100


def _calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculate simple moving average."""
    return series.rolling(window=window, min_periods=max(1, window // 2)).mean()


def _calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    # Handle edge cases:
    # - If avg_loss is 0 and avg_gain > 0: RSI = 100 (all gains, no losses)
    # - If avg_gain is 0 and avg_loss > 0: RSI = 0 (all losses, no gains)
    # - If both are 0: RSI = 50 (neutral)
    rsi = pd.Series(50.0, index=series.index)
    
    # Where we have losses, calculate normally
    has_loss = avg_loss > 0
    rs = avg_gain[has_loss] / avg_loss[has_loss]
    rsi[has_loss] = 100 - (100 / (1 + rs))
    
    # Where no losses but gains exist: RSI = 100
    no_loss_with_gain = (~has_loss) & (avg_gain > 0)
    rsi[no_loss_with_gain] = 100.0
    
    # Where no gains but losses exist: RSI = 0
    no_gain_with_loss = (~(avg_gain > 0)) & has_loss
    rsi[no_gain_with_loss] = 0.0
    
    return rsi


def _calculate_52w_features(group: pd.DataFrame) -> pd.DataFrame:
    """Calculate 52-week high/low related features."""
    close = group[CLOSE]
    
    # Rolling 252-day (approx 1 year trading days) high and low
    rolling_high = close.rolling(window=252, min_periods=20).max()
    rolling_low = close.rolling(window=252, min_periods=20).min()
    
    # Percentage from 52-week high/low
    group[PCT_FROM_52W_HIGH] = ((close - rolling_high) / rolling_high) * 100
    group[PCT_FROM_52W_LOW] = ((close - rolling_low) / rolling_low) * 100
    
    # Days since 52-week high/low (using argmax/argmin on rolling window)
    # Simplified: just use the ratio as a proxy
    group[DAYS_SINCE_52W_HIGH] = (rolling_high / close - 1) * 252  # Rough proxy
    group[DAYS_SINCE_52W_LOW] = (close / rolling_low - 1) * 252  # Rough proxy
    
    return group


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical/price-based features to the DataFrame.
    
    All features are calculated per-ticker to avoid leakage.
    Uses only past data (rolling windows look backward).
    
    Args:
        df: Wide format DataFrame with OHLCV columns.
    
    Returns:
        DataFrame with additional technical features.
    """
    if df.empty or CLOSE not in df.columns:
        return df
    
    # Sort for proper rolling calculations
    df = df.sort_values([TICKER, TIMESTAMP])
    
    result_dfs = []
    
    for ticker in df[TICKER].unique():
        ticker_df = df[df[TICKER] == ticker].copy()
        ticker_df = ticker_df.sort_values(TIMESTAMP).reset_index(drop=True)
        
        close = ticker_df[CLOSE]
        
        # Core returns (reduced set)
        ticker_df[RETURN_5D] = _calculate_returns(close, 5)
        ticker_df[RETURN_20D] = _calculate_returns(close, 20)
        ticker_df[RETURN_60D] = _calculate_returns(close, 60)
        
        # Core volatility
        ticker_df[VOLATILITY_20D] = _calculate_volatility(close, 20)
        
        # Key SMAs
        ticker_df[SMA_20] = _calculate_sma(close, 20)
        ticker_df[SMA_60] = _calculate_sma(close, 60)
        
        # Price relative to SMA (momentum)
        ticker_df[PRICE_TO_SMA_20] = (close / ticker_df[SMA_20] - 1) * 100
        ticker_df[PRICE_TO_SMA_60] = (close / ticker_df[SMA_60] - 1) * 100
        
        # RSI
        ticker_df[RSI_14] = _calculate_rsi(close, 14)
        
        # 52-week features (simplified)
        rolling_high = close.rolling(window=252, min_periods=20).max()
        rolling_low = close.rolling(window=252, min_periods=20).min()
        ticker_df[PCT_FROM_52W_HIGH] = ((close - rolling_high) / rolling_high) * 100
        ticker_df[PCT_FROM_52W_LOW] = ((close - rolling_low) / rolling_low) * 100
        
        result_dfs.append(ticker_df)
    
    if not result_dfs:
        return df
    
    result = pd.concat(result_dfs, ignore_index=True)
    
    # Convert new float64 columns to float32 to save memory
    for col in [RETURN_5D, RETURN_20D, RETURN_60D, VOLATILITY_20D, 
                SMA_20, SMA_60, PRICE_TO_SMA_20, PRICE_TO_SMA_60, 
                RSI_14, PCT_FROM_52W_HIGH, PCT_FROM_52W_LOW]:
        if col in result.columns:
            result[col] = result[col].astype('float32')
    
    return result


def get_technical_feature_columns() -> list[str]:
    """Return list of technical feature column names."""
    return [
        RETURN_1D, RETURN_5D, RETURN_20D, RETURN_60D, RETURN_252D,
        VOLATILITY_5D, VOLATILITY_20D, VOLATILITY_60D,
        SMA_5, SMA_20, SMA_60, SMA_200,
        PRICE_TO_SMA_20, PRICE_TO_SMA_60, PRICE_TO_SMA_200,
        SMA_20_TO_SMA_60, SMA_60_TO_SMA_200,
        RSI_14, HIGH_LOW_RANGE, CLOSE_TO_HIGH, CLOSE_TO_LOW,
        VOLUME_SMA_20, VOLUME_RATIO,
        MOMENTUM_5D, MOMENTUM_20D,
        DAYS_SINCE_52W_HIGH, DAYS_SINCE_52W_LOW,
        PCT_FROM_52W_HIGH, PCT_FROM_52W_LOW,
    ]
