"""Advanced financial features based on quantitative finance research.

These features are derived from academic papers on stock prediction:
- Average True Range (ATR) for volatility
- Average Directional Index (ADX) for trend strength
- Stochastic Oscillator for momentum
- On-Balance Volume (OBV) for volume-price relationship
- Volume-Weighted Average Price (VWAP) deviation
- Price patterns (gaps, inside days, etc.)
"""

import pandas as pd
import numpy as np

from config.column_names import TIMESTAMP, TICKER, CLOSE, OPEN, HIGH, LOW, VOLUME


# Feature name constants
ATR_14 = "atr_14"
ADX_14 = "adx_14"
PLUS_DI = "plus_di_14"
MINUS_DI = "minus_di_14"
STOCH_K = "stoch_k"
STOCH_D = "stoch_d"
OBV_CHANGE = "obv_change_20d"
VWAP_DEVIATION = "vwap_deviation"
GAP_PCT = "gap_pct"
BODY_TO_RANGE = "body_to_range"
UPPER_SHADOW = "upper_shadow_pct"
LOWER_SHADOW = "lower_shadow_pct"
INSIDE_DAY = "inside_day"
OUTSIDE_DAY = "outside_day"
VOLUME_MOMENTUM = "volume_momentum"
PRICE_MOMENTUM_ACCEL = "price_momentum_accel"


def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    
    return atr


def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Average Directional Index and +DI/-DI."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    
    # +DM and -DM
    plus_dm = high - prev_high
    minus_dm = prev_low - low
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed TR, +DM, -DM using Wilder smoothing
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    
    # +DI and -DI
    plus_di = 100 * smooth_plus_dm / atr.replace(0, np.nan)
    minus_di = 100 * smooth_minus_dm / atr.replace(0, np.nan)
    
    # DX and ADX
    di_diff = (plus_di - minus_di).abs()
    di_sum = plus_di + minus_di
    dx = 100 * di_diff / di_sum.replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx, plus_di, minus_di


def _calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                          k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator %K and %D."""
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    
    denom = highest_high - lowest_low
    stoch_k = 100 * (close - lowest_low) / denom.replace(0, np.nan)
    stoch_d = stoch_k.rolling(window=d_period, min_periods=1).mean()
    
    return stoch_k, stoch_d


def _calculate_obv_change(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Calculate On-Balance Volume change over period."""
    close_diff = close.diff()
    direction = np.sign(close_diff)
    obv = (volume * direction).cumsum()
    obv_change = obv.pct_change(periods=period, fill_method=None) * 100
    return obv_change


def _calculate_vwap_deviation(high: pd.Series, low: pd.Series, close: pd.Series, 
                               volume: pd.Series, period: int = 20) -> pd.Series:
    """Calculate deviation from rolling VWAP."""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)
    deviation = ((close - vwap) / vwap) * 100
    return deviation


def _calculate_candlestick_features(open_: pd.Series, high: pd.Series, 
                                     low: pd.Series, close: pd.Series) -> dict[str, pd.Series]:
    """Calculate candlestick pattern features."""
    # Gap percentage (open vs previous close)
    prev_close = close.shift(1)
    gap_pct = ((open_ - prev_close) / prev_close.replace(0, np.nan)) * 100
    
    # Body to range ratio (0-1, higher = more decisive move)
    body = (close - open_).abs()
    range_ = (high - low).replace(0, np.nan)
    body_to_range = body / range_
    
    # Shadow percentages
    upper_shadow = (high - pd.concat([open_, close], axis=1).max(axis=1)) / range_
    lower_shadow = (pd.concat([open_, close], axis=1).min(axis=1) - low) / range_
    
    # Inside day (today's range within yesterday's range)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    inside_day = ((high <= prev_high) & (low >= prev_low)).astype('float32')
    
    # Outside day (today's range encompasses yesterday's range)
    outside_day = ((high >= prev_high) & (low <= prev_low)).astype('float32')
    
    return {
        GAP_PCT: gap_pct,
        BODY_TO_RANGE: body_to_range,
        UPPER_SHADOW: upper_shadow,
        LOWER_SHADOW: lower_shadow,
        INSIDE_DAY: inside_day,
        OUTSIDE_DAY: outside_day,
    }


def _calculate_momentum_features(close: pd.Series, volume: pd.Series) -> dict[str, pd.Series]:
    """Calculate momentum-related features."""
    # Volume momentum (current volume vs 20-day avg)
    vol_sma = volume.rolling(20, min_periods=1).mean()
    vol_momentum = volume / vol_sma.replace(0, np.nan)
    
    # Price momentum acceleration (rate of change of momentum)
    mom_5 = close.pct_change(5)
    mom_20 = close.pct_change(20)
    accel = mom_5 - mom_5.shift(5)  # Change in momentum
    
    return {
        VOLUME_MOMENTUM: vol_momentum,
        PRICE_MOMENTUM_ACCEL: accel * 100,
    }


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced financial features to the DataFrame.
    
    All features are calculated per-ticker to avoid leakage.
    
    Args:
        df: Wide format DataFrame with OHLCV columns.
    
    Returns:
        DataFrame with additional advanced features.
    """
    required_cols = {CLOSE, OPEN, HIGH, LOW}
    if df.empty or not required_cols.issubset(df.columns):
        return df
    
    has_volume = VOLUME in df.columns
    
    df = df.sort_values([TICKER, TIMESTAMP])
    result_dfs = []
    
    for ticker in df[TICKER].unique():
        ticker_df = df[df[TICKER] == ticker].copy()
        ticker_df = ticker_df.sort_values(TIMESTAMP).reset_index(drop=True)
        
        high = ticker_df[HIGH]
        low = ticker_df[LOW]
        close = ticker_df[CLOSE]
        open_ = ticker_df[OPEN]
        volume = ticker_df[VOLUME] if has_volume else pd.Series(1, index=ticker_df.index)
        
        # ATR
        ticker_df[ATR_14] = _calculate_atr(high, low, close)
        
        # ADX and directional indicators
        ticker_df[ADX_14], ticker_df[PLUS_DI], ticker_df[MINUS_DI] = _calculate_adx(high, low, close)
        
        # Stochastic Oscillator
        ticker_df[STOCH_K], ticker_df[STOCH_D] = _calculate_stochastic(high, low, close)
        
        # Volume-based features
        if has_volume:
            ticker_df[OBV_CHANGE] = _calculate_obv_change(close, volume)
            ticker_df[VWAP_DEVIATION] = _calculate_vwap_deviation(high, low, close, volume)
        
        # Candlestick features
        candle_features = _calculate_candlestick_features(open_, high, low, close)
        for name, series in candle_features.items():
            ticker_df[name] = series
        
        # Momentum features
        mom_features = _calculate_momentum_features(close, volume)
        for name, series in mom_features.items():
            ticker_df[name] = series
        
        result_dfs.append(ticker_df)
    
    if not result_dfs:
        return df
    
    result = pd.concat(result_dfs, ignore_index=True)
    
    # Convert to float32 to save memory and handle infinities
    float_cols = [ATR_14, ADX_14, PLUS_DI, MINUS_DI, STOCH_K, STOCH_D,
                  OBV_CHANGE, VWAP_DEVIATION, GAP_PCT, BODY_TO_RANGE,
                  UPPER_SHADOW, LOWER_SHADOW, VOLUME_MOMENTUM, PRICE_MOMENTUM_ACCEL]
    
    for col in float_cols:
        if col in result.columns:
            # Replace infinities with NaN, then convert
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)
            result[col] = result[col].astype('float32')
    
    return result


def get_advanced_feature_columns() -> list[str]:
    """Return list of advanced feature column names."""
    return [
        ATR_14, ADX_14, PLUS_DI, MINUS_DI,
        STOCH_K, STOCH_D,
        OBV_CHANGE, VWAP_DEVIATION,
        GAP_PCT, BODY_TO_RANGE, UPPER_SHADOW, LOWER_SHADOW,
        INSIDE_DAY, OUTSIDE_DAY,
        VOLUME_MOMENTUM, PRICE_MOMENTUM_ACCEL,
    ]
