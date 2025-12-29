"""Alpha factor features based on academic finance research.

This module implements well-documented alpha factors from academic literature:

1. **Short-term Reversal** (Jegadeesh 1990):
   - Stocks that drop in the short term tend to bounce back
   - 1-week returns predict negative returns

2. **Momentum Quality** (momentum with trend consistency):
   - R-squared of price trend measures momentum quality
   - High R² momentum is more predictive

3. **Idiosyncratic Volatility** (Ang et al. 2006):
   - Residual volatility after removing market factor
   - Low idiosyncratic vol stocks tend to outperform (volatility anomaly)

4. **Information Discreteness** (Da, Gurun, Warachka 2014):
   - Momentum from a few large moves vs many small moves
   - Continuous information (many small moves) has more persistence

5. **Maximum Returns** (Bali, Cakici, Whitelaw 2011):
   - Stocks with extreme recent returns tend to underperform
   - MAX effect: lottery-like stocks are overpriced

6. **Skewness & Kurtosis** (higher moment features):
   - Investors overpay for positive skewness
   - Fat tails indicate regime-switching behavior

7. **Volume Patterns** (turnover, Amihud illiquidity):
   - High turnover may signal informed trading
   - Illiquidity premium (Amihud 2002)

References:
- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns.
- Ang, A., Hodrick, R.J., Xing, Y., Zhang, X. (2006). The cross-section of volatility and expected returns.
- Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects.
- Gu, S., Kelly, B., Xiu, D. (2020). Empirical Asset Pricing via Machine Learning.
"""

import pandas as pd
import numpy as np
from typing import Optional

from config.columns import CLOSE, HIGH, LOW, OPEN, VOLUME, TICKER, TIMESTAMP
from config.settings import EPSILON


def add_alpha_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add research-backed alpha factors to the DataFrame.
    
    Features are computed per-ticker to avoid mixing data across assets.
    
    Args:
        df: Wide format DataFrame with OHLCV columns.
    
    Returns:
        DataFrame with alpha factor features added.
    """
    if TICKER not in df.columns:
        return df
    
    df = df.sort_values([TICKER, TIMESTAMP])
    result_dfs = []
    
    for ticker in df[TICKER].unique():
        ticker_df = df[df[TICKER] == ticker].copy()
        
        # Skip if not enough data (need at least 1 year)
        if len(ticker_df) < 260:
            result_dfs.append(ticker_df)
            continue
        
        # --- Short-term Reversal (Jegadeesh 1990) ---
        ticker_df = _add_reversal_features(ticker_df)
        
        # --- Momentum Quality (trend R-squared) ---
        ticker_df = _add_momentum_quality(ticker_df)
        
        # --- Idiosyncratic Volatility ---
        ticker_df = _add_idiosyncratic_volatility(ticker_df)
        
        # --- Information Discreteness ---
        ticker_df = _add_information_discreteness(ticker_df)
        
        # --- Maximum Returns (lottery effect) ---
        ticker_df = _add_max_returns(ticker_df)
        
        # --- Higher Moments (skewness, kurtosis) ---
        ticker_df = _add_higher_moments(ticker_df)
        
        # --- Volume Patterns ---
        ticker_df = _add_volume_features(ticker_df)
        ticker_df = _add_amihud_illiquidity(ticker_df, window=20)
        
        # --- Price Acceleration ---
        ticker_df = _add_momentum_acceleration(ticker_df)
        
        # --- Long-term Momentum Consistency (for 365-day horizon) ---
        ticker_df = _add_long_term_momentum_consistency(ticker_df)
        
        result_dfs.append(ticker_df)
    
    if not result_dfs:
        return df
    
    return pd.concat(result_dfs).sort_values(TIMESTAMP).reset_index(drop=True)


def _add_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add short-term reversal features.
    
    Short-term reversal: stocks that dropped in the past week tend to bounce.
    This is different from momentum (which is positive autocorrelation over months).
    """
    if CLOSE not in df.columns:
        return df
    
    returns = df[CLOSE].pct_change()
    
    # 1-week return (5 trading days) - classic reversal window
    df["Rev_5d"] = df[CLOSE].pct_change(5)
    
    # 2-week return - extended reversal
    df["Rev_10d"] = df[CLOSE].pct_change(10)
    
    # Weekly return excluding most recent day (skip-day reversal)
    # This separates the reversal signal from microstructure noise
    df["Rev_5d_Skip1"] = df[CLOSE].shift(1).pct_change(5)
    
    return df


def _add_momentum_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum quality features based on trend consistency.
    
    High R² of price trend indicates consistent, high-quality momentum.
    Low R² suggests noisy price movement that may not persist.
    """
    if CLOSE not in df.columns:
        return df
    
    # Rolling trend R² for different windows
    for window in [20, 60, 120]:
        r_squared = _compute_rolling_rsquared(df[CLOSE], window)
        df[f"Trend_RSq_{window}"] = r_squared
        
        # Momentum × R² interaction (quality-adjusted momentum)
        mom = df[CLOSE].pct_change(window)
        df[f"QualMom_{window}"] = mom * r_squared
    
    return df


def _compute_rolling_rsquared(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling R-squared of linear trend.
    
    Uses the formula: R² = 1 - SSE/SST
    where SSE = sum of squared residuals from linear fit
    and SST = total sum of squares
    """
    result = pd.Series(index=series.index, dtype=float)
    values = series.values
    
    for i in range(window - 1, len(values)):
        y = values[i - window + 1:i + 1]
        
        # Skip if all NaN or insufficient variation
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < window // 2:
            result.iloc[i] = np.nan
            continue
        
        y_valid = y[valid_mask]
        x = np.arange(len(y_valid))
        
        # Simple linear regression
        x_mean = x.mean()
        y_mean = y_valid.mean()
        
        ss_tot = np.sum((y_valid - y_mean) ** 2)
        if ss_tot < EPSILON:
            result.iloc[i] = 0.0
            continue
        
        ss_xy = np.sum((x - x_mean) * (y_valid - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)
        
        if ss_xx < EPSILON:
            result.iloc[i] = 0.0
            continue
        
        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        
        y_pred = slope * x + intercept
        ss_res = np.sum((y_valid - y_pred) ** 2)
        
        r_squared = 1.0 - (ss_res / ss_tot)
        result.iloc[i] = max(0.0, min(1.0, r_squared))  # Clip to [0, 1]
    
    return result


def _add_idiosyncratic_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Add idiosyncratic volatility features.
    
    Since we don't have market returns in this context, we use:
    1. Residual volatility from AR(1) model
    2. Volatility of returns orthogonal to rolling mean
    
    The low volatility anomaly: low idiosyncratic vol stocks outperform.
    """
    if CLOSE not in df.columns:
        return df
    
    returns = df[CLOSE].pct_change()
    
    # Rolling residual volatility (volatility after removing trend)
    for window in [20, 60]:
        rolling_mean = returns.rolling(window).mean()
        residuals = returns - rolling_mean
        df[f"IdioVol_{window}"] = residuals.rolling(window).std()
    
    # Volatility of volatility (vol clustering measure)
    vol_20 = returns.rolling(20).std()
    df["VolOfVol_60"] = vol_20.rolling(60).std()
    
    return df


def _add_information_discreteness(df: pd.DataFrame) -> pd.DataFrame:
    """Add information discreteness feature.
    
    Measures whether momentum comes from many small moves (continuous)
    or a few large moves (discrete).
    
    Formula: sign(sum of returns) × (# positive days - # negative days) / total days
    
    Continuous information momentum (high discreteness score) tends to persist.
    """
    if CLOSE not in df.columns:
        return df
    
    returns = df[CLOSE].pct_change()
    
    for window in [21, 63, 126]:  # 1 month, 3 months, 6 months
        # Count positive and negative days
        pos_days = returns.rolling(window).apply(lambda x: (x > 0).sum(), raw=True)
        neg_days = returns.rolling(window).apply(lambda x: (x < 0).sum(), raw=True)
        
        # Sum of returns (momentum direction)
        ret_sum = returns.rolling(window).sum()
        sign_ret = np.sign(ret_sum)
        
        # Information discreteness
        df[f"InfoDisc_{window}"] = sign_ret * (pos_days - neg_days) / window
    
    return df


def _add_max_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add maximum return features.
    
    The MAX effect (Bali, Cakici, Whitelaw 2011):
    Stocks with extreme positive returns in the past month underperform.
    Investors overpay for lottery-like characteristics.
    """
    if CLOSE not in df.columns:
        return df
    
    returns = df[CLOSE].pct_change()
    
    # Maximum daily return in past month
    df["MAX_21d"] = returns.rolling(21).max()
    
    # Minimum daily return (maximum loss)
    df["MIN_21d"] = returns.rolling(21).min()
    
    # Average of top 5 returns in past month (robust MAX)
    df["MAX5_21d"] = returns.rolling(21).apply(
        lambda x: np.sort(x)[-5:].mean() if len(x) >= 5 else np.nan, raw=True
    )
    
    # Max - Min spread (extreme move range)
    df["MaxMinSpread_21d"] = df["MAX_21d"] - df["MIN_21d"]
    
    return df


def _add_higher_moments(df: pd.DataFrame) -> pd.DataFrame:
    """Add return distribution higher moments.
    
    - Skewness: investors overpay for positive skew (lottery preference)
    - Kurtosis: fat tails indicate regime switching, crash risk
    """
    if CLOSE not in df.columns:
        return df
    
    returns = df[CLOSE].pct_change()
    
    # Rolling skewness (60 days = 3 months)
    df["Skew_60d"] = returns.rolling(60).skew()
    df["Skew_126d"] = returns.rolling(126).skew()
    
    # Rolling kurtosis (excess kurtosis)
    df["Kurt_60d"] = returns.rolling(60).kurt()
    df["Kurt_126d"] = returns.rolling(126).kurt()
    
    # Downside deviation (semi-variance, only negative returns)
    df["DownVol_60d"] = returns.rolling(60).apply(
        lambda x: np.sqrt(np.mean(np.minimum(x, 0) ** 2)), raw=True
    )
    
    # Upside/downside volatility ratio
    upside_vol = returns.rolling(60).apply(
        lambda x: np.sqrt(np.mean(np.maximum(x, 0) ** 2) + EPSILON), raw=True
    )
    df["UpDownRatio_60d"] = upside_vol / (df["DownVol_60d"] + EPSILON)
    
    return df


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based features.
    
    - Turnover (volume relative to average): signals informed trading
    - Amihud illiquidity: price impact per dollar volume
    - Volume-return correlation: may indicate informed trading
    """
    if VOLUME not in df.columns or CLOSE not in df.columns:
        return df
    
    returns = df[CLOSE].pct_change()
    
    # Relative volume (current vs rolling average)
    vol_ma_20 = df[VOLUME].rolling(20).mean()
    vol_ma_60 = df[VOLUME].rolling(60).mean()
    
    df["RelVol_20d"] = df[VOLUME] / (vol_ma_20 + EPSILON)
    df["RelVol_60d"] = df[VOLUME] / (vol_ma_60 + EPSILON)
    
    # Volume momentum (change in average volume)
    df["VolMom_20d"] = vol_ma_20.pct_change(20)
    
    # Amihud illiquidity (|return| / dollar volume)
    # Higher = more illiquid = higher expected returns (illiquidity premium)
    dollar_volume = df[VOLUME] * df[CLOSE]
    daily_illiq = returns.abs() / (dollar_volume + EPSILON)
    
    df["Amihud_21d"] = daily_illiq.rolling(21).mean()
    df["Amihud_63d"] = daily_illiq.rolling(63).mean()
    
    # Volume-return correlation (informed trading signal)
    df["VolRetCorr_60d"] = returns.rolling(60).corr(df[VOLUME])
    
    # Absolute volume-return correlation (more robust)
    df["AbsVolRetCorr_60d"] = returns.abs().rolling(60).corr(df[VOLUME])
    
    return df


def _add_momentum_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum acceleration features.
    
    Measures whether momentum is accelerating or decelerating.
    Accelerating momentum may have more predictive power.
    """
    if CLOSE not in df.columns:
        return df
    
    # Short-term and long-term momentum
    mom_21 = df[CLOSE].pct_change(21)
    mom_63 = df[CLOSE].pct_change(63)
    mom_126 = df[CLOSE].pct_change(126)
    
    # Momentum acceleration: recent vs older momentum
    df["MomAccel_21_63"] = mom_21 - mom_63 / 3  # Normalize for time
    df["MomAccel_63_126"] = mom_63 - mom_126 / 2
    
    # Momentum consistency: correlation of returns over time
    returns = df[CLOSE].pct_change()
    
    # First half vs second half of period
    first_half_ret = returns.rolling(30).apply(
        lambda x: x[:15].sum() if len(x) >= 30 else np.nan, raw=True
    )
    second_half_ret = returns.rolling(30).apply(
        lambda x: x[15:].sum() if len(x) >= 30 else np.nan, raw=True
    )
    df["MomConsist_60d"] = first_half_ret * second_half_ret
    
    # 52-week high momentum (Geo & Tian 2008)
    # Distance from 52-week high predicts continuation
    if HIGH in df.columns:
        high_52w = df[HIGH].rolling(252).max()
        df["Near52wHigh"] = df[CLOSE] / (high_52w + EPSILON)
    
    return df


def _add_long_term_momentum_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Add long-term momentum consistency features for annual horizon prediction.
    
    Based on research showing that consistent momentum across multiple periods
    is more predictive than raw momentum for longer horizons (Moskowitz et al. 2012).
    
    These features are specifically designed for 365-day forward return prediction.
    """
    if CLOSE not in df.columns:
        return df
    
    returns = df[CLOSE].pct_change()
    
    # Quarterly momentum: consistent positive quarters predict annual returns
    # Count of positive quarter returns in past year
    q1_ret = returns.rolling(63).sum()   # ~3 months
    q2_ret = returns.shift(63).rolling(63).sum()
    q3_ret = returns.shift(126).rolling(63).sum()
    q4_ret = returns.shift(189).rolling(63).sum()
    
    # Number of positive quarters (0-4)
    df["PosQuarters_252d"] = (
        (q1_ret > 0).astype(int) + 
        (q2_ret > 0).astype(int) + 
        (q3_ret > 0).astype(int) + 
        (q4_ret > 0).astype(int)
    )
    
    # Average quarterly return (more stable than annual)
    df["AvgQuarterRet_252d"] = (q1_ret + q2_ret + q3_ret + q4_ret) / 4
    
    # Consistency score: std of quarterly returns (lower = more consistent)
    quarterly_rets = pd.concat([q1_ret, q2_ret, q3_ret, q4_ret], axis=1)
    df["MomConsist_252d"] = quarterly_rets.std(axis=1)
    
    # Momentum quality: annual return / consistency (higher is better)
    annual_ret = df[CLOSE].pct_change(252)
    df["MomQuality_252d"] = annual_ret / (df["MomConsist_252d"] + EPSILON)
    
    # Trend persistence: autocorrelation of monthly returns over past year
    monthly_rets = returns.rolling(21).sum()
    df["TrendPersist_252d"] = monthly_rets.rolling(252).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 252 else np.nan,
        raw=False
    )
    
    # Long-term vs short-term momentum divergence
    # Positive = recent acceleration, negative = deceleration
    mom_126 = df[CLOSE].pct_change(126)
    mom_252 = df[CLOSE].pct_change(252)
    df["MomDivergence_126_252"] = mom_126 - (mom_252 / 2)
    
    return df


def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add seasonality and calendar features.
    
    Known calendar anomalies:
    - January effect
    - Turn of month effect
    - Day of week effects
    
    Note: This requires datetime conversion from timestamp.
    """
    if TIMESTAMP not in df.columns:
        return df
    
    # Convert timestamp to datetime
    dt = pd.to_datetime(df[TIMESTAMP], unit='ms')
    
    # Month of year (January = 1, December = 12)
    df["Month"] = dt.dt.month
    
    # Day of month (turn of month effect)
    df["DayOfMonth"] = dt.dt.day
    df["IsMonthEnd"] = (dt.dt.is_month_end).astype(int)
    df["IsMonthStart"] = (dt.dt.is_month_start).astype(int)
    
    # Turn of month (last 3 and first 3 days)
    df["TurnOfMonth"] = ((dt.dt.day <= 3) | (dt.dt.day >= 28)).astype(int)
    
    # Quarter (for quarterly earnings effects)
    df["Quarter"] = dt.dt.quarter
    
    # Day of week (Monday = 0, Friday = 4)
    df["DayOfWeek"] = dt.dt.dayofweek
    
    # Is Monday (Monday effect - historically negative)
    df["IsMonday"] = (dt.dt.dayofweek == 0).astype(int)
    
    # Is Friday (Friday effect - sentiment before weekend)
    df["IsFriday"] = (dt.dt.dayofweek == 4).astype(int)
    
    return df


def _add_amihud_illiquidity(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate Amihud Illiquidity (Abs(Ret) / Volume)."""
    if VOLUME not in df.columns or CLOSE not in df.columns:
        return df
        
    # Avoid division by zero
    illiq = df[CLOSE].pct_change().abs() / (df[VOLUME] * df[CLOSE] + EPSILON)
    df[f"Amihud_{window}"] = illiq.rolling(window=window).mean()
    return df
