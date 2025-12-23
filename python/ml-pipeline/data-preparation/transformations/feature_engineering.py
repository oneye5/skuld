"""Module for feature engineering - adding derived features."""

import pandas as pd
import numpy as np

from config.column_names import (
    TIMESTAMP,
    DAY_OF_YEAR_SIN,
    DAY_OF_YEAR_COS,
    DAY_OF_WEEK_SIN,
    DAY_OF_WEEK_COS,
    MONTH_SIN,
    MONTH_COS,
    # Financial data columns
    ANNUAL_NET_INCOME,
    ANNUAL_BASIC_AVERAGE_SHARE,
    ANNUAL_TOTAL_REVENUE,
    ANNUAL_SGA,
    ANNUAL_DEPRECIATION,
    TRAILING_FEES_COMMISSION,
    LONG_TERM_INTEREST_RATE,
    IMMEDIATE_INTEREST_RATE,
    SHORT_TERM_INTEREST_RATE,
    # Engineered ratio features
    EPS_BASIC,
    NET_PROFIT_MARGIN,
    SGA_RATIO,
    DEPRECIATION_RATIO,
    COMMISSION_EFFICIENCY,
    IMMEDIATE_INTEREST_VOLATILITY,
    SHORT_TERM_INTEREST_VOLATILITY,
)


def _safe_divide(df: pd.DataFrame, numerator: str, denominator: str, epsilon: float = 1e-6) -> pd.Series | None:
    """
    Safely divide two columns, returning None if either column is missing.
    
    Args:
        df: DataFrame containing the columns.
        numerator: Name of the numerator column.
        denominator: Name of the denominator column.
        epsilon: Small value to avoid division by zero.
    
    Returns:
        Series with division result, or None if columns are missing.
    """
    if numerator not in df.columns or denominator not in df.columns:
        return None
    return df[numerator] / (df[denominator] + epsilon)


def _safe_subtract(df: pd.DataFrame, col1: str, col2: str) -> pd.Series | None:
    """
    Safely subtract two columns, returning None if either column is missing.
    
    Args:
        df: DataFrame containing the columns.
        col1: Name of the first column.
        col2: Name of the column to subtract.
    
    Returns:
        Series with subtraction result, or None if columns are missing.
    """
    if col1 not in df.columns or col2 not in df.columns:
        return None
    return df[col1] - df[col2]


def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add financial ratio features derived from fundamental data.
    
    These ratios capture relative relationships between financial metrics
    that would otherwise be lost after scaling. Following nzx-predictor approach.
    
    Args:
        df: DataFrame with financial data columns.
    
    Returns:
        DataFrame with added ratio features.
    """
    new_features = {}
    
    # EPS Basic = Net Income / Shares Outstanding
    result = _safe_divide(df, ANNUAL_NET_INCOME, ANNUAL_BASIC_AVERAGE_SHARE)
    if result is not None:
        new_features[EPS_BASIC] = result
    
    # Net Profit Margin = Net Income / Revenue
    result = _safe_divide(df, ANNUAL_NET_INCOME, ANNUAL_TOTAL_REVENUE)
    if result is not None:
        new_features[NET_PROFIT_MARGIN] = result
    
    # SG&A Ratio = SG&A Expense / Revenue
    result = _safe_divide(df, ANNUAL_SGA, ANNUAL_TOTAL_REVENUE)
    if result is not None:
        new_features[SGA_RATIO] = result
    
    # Depreciation Ratio = Depreciation / Revenue
    result = _safe_divide(df, ANNUAL_DEPRECIATION, ANNUAL_TOTAL_REVENUE)
    if result is not None:
        new_features[DEPRECIATION_RATIO] = result
    
    # Commission Efficiency = Commission Expense / Revenue
    result = _safe_divide(df, TRAILING_FEES_COMMISSION, ANNUAL_TOTAL_REVENUE)
    if result is not None:
        new_features[COMMISSION_EFFICIENCY] = result
    
    # Interest rate spreads (volatility indicators)
    result = _safe_subtract(df, LONG_TERM_INTEREST_RATE, IMMEDIATE_INTEREST_RATE)
    if result is not None:
        new_features[IMMEDIATE_INTEREST_VOLATILITY] = result
    
    result = _safe_subtract(df, LONG_TERM_INTEREST_RATE, SHORT_TERM_INTEREST_RATE)
    if result is not None:
        new_features[SHORT_TERM_INTEREST_VOLATILITY] = result
    
    if new_features:
        features_df = pd.DataFrame(new_features, index=df.index)
        return pd.concat([df, features_df], axis=1)
    
    return df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical time-based features derived from timestamp.

    Cyclical encoding using sin/cos ensures continuity (e.g., Dec 31 is 
    close to Jan 1).

    Args:
        df: DataFrame with timestamp column (Unix timestamp in milliseconds).

    Returns:
        DataFrame with added cyclical time features.
    """
    # Convert timestamp to datetime
    dt = pd.to_datetime(df[TIMESTAMP], unit='ms')

    # Day of year (1-366)
    day_of_year = dt.dt.dayofyear
    # Day of week (0-6)
    day_of_week = dt.dt.dayofweek
    # Month (1-12)
    month = dt.dt.month

    # Create all features at once to avoid fragmentation
    new_features = pd.DataFrame({
        DAY_OF_YEAR_SIN: np.sin(2 * np.pi * day_of_year / 365.25),
        DAY_OF_YEAR_COS: np.cos(2 * np.pi * day_of_year / 365.25),
        DAY_OF_WEEK_SIN: np.sin(2 * np.pi * day_of_week / 7),
        DAY_OF_WEEK_COS: np.cos(2 * np.pi * day_of_week / 7),
        MONTH_SIN: np.sin(2 * np.pi * month / 12),
        MONTH_COS: np.cos(2 * np.pi * month / 12),
    }, index=df.index)

    return pd.concat([df, new_features], axis=1)
