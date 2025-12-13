"""Technical analysis features for momentum, trend, and volatility detection.

Creates indicators that help:
1. Identify market momentum (RSI, MACD, momentum)
2. Detect trends (EMA, price position in range)
3. Quantify volatility (ATR, Bollinger Bands)
4. Avoid value traps (avoid buying falling stocks)

All features computed within ticker groups to prevent data leakage.
Uses 20-50 period lookback windows (configurable).
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List


class TechnicalFeatures:
    """Technical indicator computation with leakage prevention."""
    
    LOOKBACK_RSI = 14  # RSI period
    LOOKBACK_MACD_FAST = 12  # MACD fast EMA
    LOOKBACK_MACD_SLOW = 26  # MACD slow EMA
    LOOKBACK_MACD_SIGNAL = 9  # MACD signal line
    LOOKBACK_BB = 20  # Bollinger Bands period
    LOOKBACK_ATR = 14  # ATR period
    LOOKBACK_EMA = 50  # Long-term trend EMA
    LOOKBACK_MOMENTUM = 10  # Momentum window
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index (0-100).
        
        Measures momentum. RSI > 70 = overbought, RSI < 30 = oversold.
        Helps avoid buying already-rallied stocks (avoid momentum crashes).
        
        Args:
            close: Series of closing prices.
            period: Lookback period (default 14).
        
        Returns:
            Series of RSI values (0-100).
        """
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(close: pd.Series, 
             fast: int = 12, 
             slow: int = 26, 
             signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence).
        
        Captures trend strength and reversal signals.
        - MACD > Signal = bullish (buying)
        - MACD < Signal = bearish (selling)
        - MACD histogram = momentum
        
        Args:
            close: Series of closing prices.
            fast: Fast EMA period.
            slow: Slow EMA period.
            signal: Signal line period.
        
        Returns:
            Tuple of (MACD, Signal, Histogram) series.
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def momentum(close: pd.Series, period: int = 10) -> pd.Series:
        """Price momentum: current price vs price N periods ago.
        
        Positive momentum = price going up (good sign).
        Negative momentum = price going down (avoid).
        
        Args:
            close: Series of closing prices.
            period: Lookback period.
        
        Returns:
            Series of momentum values (ratio, 1.0 = no change).
        """
        return close / close.shift(period)
    
    @staticmethod
    def bollinger_bands(close: pd.Series, 
                       period: int = 20, 
                       std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands: volatility-based support/resistance.
        
        - Close near upper band = overbought
        - Close near lower band = oversold
        - Narrow bands = low volatility (quiet market)
        - Wide bands = high volatility (risky)
        
        Args:
            close: Series of closing prices.
            period: Lookback period.
            std_dev: Number of standard deviations (default 2).
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band) series.
        """
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def bb_position(close: pd.Series, upper: pd.Series, lower: pd.Series) -> pd.Series:
        """Position within Bollinger Bands (0-1, where 0.5 = middle).
        
        - 0.0 = at lower band (oversold)
        - 0.5 = at middle/SMA (neutral)
        - 1.0 = at upper band (overbought)
        
        Avoids buying when overbought (position > 0.7).
        
        Args:
            close: Series of closing prices.
            upper: Upper Bollinger Band series.
            lower: Lower Bollinger Band series.
        
        Returns:
            Series of positions (0-1).
        """
        # Avoid division by zero when upper == lower (zero volatility)
        band_width = upper - lower
        band_width = band_width.replace(0, np.nan)  # Replace 0 width with NaN
        position = (close - lower) / band_width
        position = position.fillna(0.5)  # When zero volatility, default to middle
        return position.clip(0, 1)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> pd.Series:
        """Average True Range: volatility measure.
        
        High ATR = high volatility = risky trade.
        Low ATR = low volatility = stable, predictable.
        
        Args:
            high: Series of high prices.
            low: Series of low prices.
            close: Series of closing prices.
            period: Lookback period.
        
        Returns:
            Series of ATR values (in price units).
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def atr_ratio(close: pd.Series, atr: pd.Series) -> pd.Series:
        """ATR as percentage of close price.
        
        Normalizes volatility by price level.
        - High ratio = volatile per unit price
        - Low ratio = stable per unit price
        
        Args:
            close: Series of closing prices.
            atr: Series of ATR values.
        
        Returns:
            Series of ATR ratios (volatility %).
        """
        # Avoid division by zero when price is 0 or NaN
        ratio = atr / close.replace(0, np.nan)
        ratio = ratio.fillna(0)  # Zero price -> zero ratio
        return ratio.clip(0, 0.5)  # Cap at 50% to avoid extremes
    
    @staticmethod
    def ema(close: pd.Series, period: int = 50) -> pd.Series:
        """Exponential Moving Average: long-term trend direction.
        
        Close > EMA = in uptrend (good for buying).
        Close < EMA = in downtrend (avoid buying).
        
        Args:
            close: Series of closing prices.
            period: Lookback period.
        
        Returns:
            Series of EMA values.
        """
        return close.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def price_position_ema(close: pd.Series, ema: pd.Series) -> pd.Series:
        """Price position relative to EMA trend line.
        
        - 1.0 = price = EMA (on trend)
        - > 1.0 = price > EMA (above trend, overbought)
        - < 1.0 = price < EMA (below trend, oversold)
        
        Avoid buying when price far above EMA (ratio > 1.1).
        
        Args:
            close: Series of closing prices.
            ema: Series of EMA values.
        
        Returns:
            Series of price/EMA ratios.
        """
        return (close / ema).clip(0.7, 1.3)  # Normalize extreme values
    
    @staticmethod
    def volatility_zscore(close: pd.Series, period: int = 20) -> pd.Series:
        """Z-score of price volatility.
        
        - High Z-score = unusually volatile
        - Low Z-score = unusually calm
        
        Avoid trading during high volatility (Z > 2).
        
        Args:
            close: Series of closing prices.
            period: Lookback period for volatility.
        
        Returns:
            Series of volatility Z-scores.
        """
        daily_returns = close.pct_change().abs()
        rolling_vol = daily_returns.rolling(window=period).std()
        vol_mean = rolling_vol.rolling(window=period).mean()
        vol_std = rolling_vol.rolling(window=period).std()
        
        zscore = (rolling_vol - vol_mean) / vol_std.replace(0, 1)
        return zscore.clip(-3, 3)  # Clip extreme values
    
    @staticmethod
    def price_change_rate(close: pd.Series, period: int = 5) -> pd.Series:
        """Rate of price change (% per period).
        
        Positive = price increasing (momentum).
        Negative = price decreasing (avoid).
        
        Args:
            close: Series of closing prices.
            period: Number of periods to look back.
        
        Returns:
            Series of price change rates (0.05 = +5%).
        """
        return close.pct_change(periods=period).clip(-0.5, 0.5)
    
    @staticmethod
    def price_momentum(close: pd.Series, period: int = 20) -> pd.Series:
        """Price momentum: % change over lookback period.
        
        Captures trend strength and direction.
        - Positive = uptrend (good)
        - Negative = downtrend (avoid)
        
        Args:
            close: Series of closing prices.
            period: Lookback period (default 20).
        
        Returns:
            Series of momentum values as percentage change.
        """
        return (close / close.shift(period) - 1).clip(-1, 1)
    
    @staticmethod
    def volatility_adjusted_return(close: pd.Series, atr: pd.Series, period: int = 20) -> pd.Series:
        """Risk-adjusted return: price change normalized by volatility.
        
        High ratio = strong trend relative to noise.
        Low ratio = weak signal in noise.
        
        Good for entry signals: prefer high volatility-adjusted returns.
        
        Args:
            close: Series of closing prices.
            atr: Series of ATR values.
            period: Lookback period for returns.
        
        Returns:
            Series of volatility-adjusted returns.
        """
        returns = close.pct_change(periods=period)
        # Avoid division by zero: use close.replace(0, NaN) then fillna(1)
        vol_normalized = (atr / close.replace(0, np.nan)).fillna(1)
        adj_return = returns / vol_normalized.replace(0, 1)
        return adj_return.clip(-10, 10)


def add_technical_features(df: pd.DataFrame, 
                          ticker_col: str = 'ticker',
                          timestamp_col: str = 'timestamp',
                          close_col: str = 'Close',
                          high_col: str = 'High',
                          low_col: str = 'Low',
                          volume_col: str = 'Volume') -> pd.DataFrame:
    """Add all technical features to DataFrame.
    
    Computes all technical indicators grouped by ticker to prevent leakage.
    Features created:
    - rsi_14: RSI momentum (0-100)
    - macd: MACD line
    - macd_signal: MACD signal line
    - macd_hist: MACD histogram
    - momentum_10: 10-period momentum ratio
    - bb_upper, bb_middle, bb_lower: Bollinger Bands
    - bb_position: Position within bands (0-1)
    - atr_14: Average true range
    - atr_ratio: ATR as % of price
    - ema_50: 50-period exponential moving average
    - price_ema_ratio: Price to EMA ratio
    - volatility_zscore: Volatility Z-score
    - price_change_5d: 5-day price change rate
    - price_momentum_20: 20-day price momentum
    - volatility_adjusted_return: Risk-adjusted return signal
    
    Args:
        df: Input DataFrame with OHLCV data.
        ticker_col: Name of ticker column.
        timestamp_col: Name of timestamp column.
        close_col: Name of close price column.
        high_col: Name of high price column.
        low_col: Name of low price column.
        volume_col: Name of volume column.
    
    Returns:
        DataFrame with technical features added.
    
    Raises:
        ValueError: If required columns are missing.
    """
    required_cols = [ticker_col, close_col, high_col, low_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Sort data for ticker group processing (no unnecessary copy)
    df = df.sort_values([ticker_col, timestamp_col]).reset_index(drop=True)
    
    # Initialize feature columns
    feature_cols = [
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'momentum_10', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_position',
        'atr_14', 'atr_ratio', 'ema_50', 'price_ema_ratio',
        'volatility_zscore', 'price_change_5d', 'price_momentum_20',
        'volatility_adjusted_return'
    ]
    for col in feature_cols:
        df[col] = np.nan
    
    # Compute features per ticker to prevent data leakage
    # Use dictionary to collect all results, then bulk assign (much faster than 20+ .loc operations)
    feature_results = {col: np.full(len(df), np.nan) for col in feature_cols}
    
    for ticker, group in df.groupby(ticker_col, sort=False):
        idx = group.index
        idx_positions = np.where(df.index.isin(idx))[0]  # Get integer positions for bulk assignment
        close_series = pd.Series(group[close_col].values, index=idx)
        high_series = pd.Series(group[high_col].values, index=idx)
        low_series = pd.Series(group[low_col].values, index=idx)
        
        # === MOMENTUM ===
        rsi = TechnicalFeatures.rsi(close_series, period=TechnicalFeatures.LOOKBACK_RSI)
        feature_results['rsi_14'][idx_positions] = rsi.values
        
        # === MACD ===
        macd, signal, hist = TechnicalFeatures.macd(
            close_series,
            fast=TechnicalFeatures.LOOKBACK_MACD_FAST,
            slow=TechnicalFeatures.LOOKBACK_MACD_SLOW,
            signal=TechnicalFeatures.LOOKBACK_MACD_SIGNAL
        )
        feature_results['macd'][idx_positions] = macd.values
        feature_results['macd_signal'][idx_positions] = signal.values
        feature_results['macd_hist'][idx_positions] = hist.values
        
        # === MOMENTUM ===
        momentum = TechnicalFeatures.momentum(close_series, period=TechnicalFeatures.LOOKBACK_MOMENTUM)
        feature_results['momentum_10'][idx_positions] = momentum.values
        
        # === BOLLINGER BANDS ===
        bb_upper, bb_middle, bb_lower = TechnicalFeatures.bollinger_bands(
            close_series, 
            period=TechnicalFeatures.LOOKBACK_BB
        )
        feature_results['bb_upper'][idx_positions] = bb_upper.values
        feature_results['bb_middle'][idx_positions] = bb_middle.values
        feature_results['bb_lower'][idx_positions] = bb_lower.values
        
        bb_pos = TechnicalFeatures.bb_position(close_series, bb_upper, bb_lower)
        feature_results['bb_position'][idx_positions] = bb_pos.values
        
        # === ATR ===
        atr = TechnicalFeatures.atr(high_series, low_series, close_series, 
                                   period=TechnicalFeatures.LOOKBACK_ATR)
        feature_results['atr_14'][idx_positions] = atr.values
        
        atr_ratio = TechnicalFeatures.atr_ratio(close_series, atr)
        feature_results['atr_ratio'][idx_positions] = atr_ratio.values
        
        # === EMA TREND ===
        ema = TechnicalFeatures.ema(close_series, period=TechnicalFeatures.LOOKBACK_EMA)
        feature_results['ema_50'][idx_positions] = ema.values
        
        price_ema_ratio = TechnicalFeatures.price_position_ema(close_series, ema)
        feature_results['price_ema_ratio'][idx_positions] = price_ema_ratio.values
        
        # === VOLATILITY ===
        vol_zscore = TechnicalFeatures.volatility_zscore(close_series, period=20)
        feature_results['volatility_zscore'][idx_positions] = vol_zscore.values
        
        # === PRICE CHANGE & MOMENTUM ===
        price_change = TechnicalFeatures.price_change_rate(close_series, period=5)
        feature_results['price_change_5d'][idx_positions] = price_change.values
        
        momentum = TechnicalFeatures.price_momentum(close_series, period=20)
        feature_results['price_momentum_20'][idx_positions] = momentum.values
        
        # === VOLATILITY-ADJUSTED RETURN ===
        vol_adj_return = TechnicalFeatures.volatility_adjusted_return(close_series, atr, period=20)
        feature_results['volatility_adjusted_return'][idx_positions] = vol_adj_return.values
    
    # Bulk assign all features at once (much faster than 20+ individual .loc calls)
    for col in feature_cols:
        df[col] = feature_results[col]
    
    # Fill remaining NaN values from rolling calculations using forward fill per ticker
    for ticker in df[ticker_col].unique():
        mask = df[ticker_col] == ticker
        for col in feature_cols:
            df.loc[mask, col] = df.loc[mask, col].bfill()
    
    return df


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.utils.csv_utils import load_csv
    from src.config.config import WIDE_CSV_PATH
    
    print("Loading data...")
    df = load_csv(WIDE_CSV_PATH)
    
    print(f"Original shape: {df.shape}")
    print("Adding technical features...")
    
    df_with_tech = add_technical_features(
        df,
        ticker_col='ticker',
        timestamp_col='timestamp',
        close_col='Close',
        high_col='High',
        low_col='Low'
    )
    
    print(f"New shape: {df_with_tech.shape}")
    print("\nNew features added:")
    tech_cols = [c for c in df_with_tech.columns if any(
        c.startswith(p) for p in ['rsi_', 'macd', 'momentum_', 'bb_', 'atr_', 
                                  'ema_', 'price_', 'volatility_']
    )]
    for col in sorted(tech_cols):
        print(f"  - {col}")
    
    print("\nSample of technical features:")
    print(df_with_tech[['ticker', 'Close'] + tech_cols[:5]].head(10))
