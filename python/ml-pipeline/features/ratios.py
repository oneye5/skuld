"""Financial ratio features from nzx-predictor.

These ratios preserve relationships between features that would be lost after scaling.
Includes both OHLCV-derived features and fundamental ratios from nzx-predictor.
"""

import pandas as pd
import numpy as np

from config.columns import (
    OPEN, HIGH, LOW, CLOSE, VOLUME,
    LONG_TERM_INTEREST_RATE,
    IMMEDIATE_INTEREST_RATE,
    SHORT_TERM_INTEREST_RATE,
    IMMEDIATE_INTEREST_VOLATILITY,
    SHORT_TERM_INTEREST_VOLATILITY,
    # Fundamental columns
    ANNUAL_NET_INCOME,
    ANNUAL_BASIC_AVG_SHARES,
    ANNUAL_TOTAL_REVENUE,
    ANNUAL_SGA,
    ANNUAL_DEPRECIATION,
    TRAILING_FEES_COMMISSION,
    # Engineered feature names
    EPS_BASIC,
    NET_PROFIT_MARGIN,
    SGA_RATIO,
    DEPRECIATION_RATIO,
    COMMISSION_EFFICIENCY,
)
from config.settings import EPSILON


def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add financial ratio features derived from price data.
    
    Since fundamental data is too sparse, we focus on OHLCV-derived features.
    Ratios preserve relative information between features that would
    otherwise be lost after scaling.
    
    Args:
        df: Wide format DataFrame with OHLCV columns.
    
    Returns:
        DataFrame with ratio features added.
    """
    result = df.copy()
    
    # Price-based ratios (always available from OHLCV)
    if all(col in result.columns for col in [OPEN, HIGH, LOW, CLOSE]):
        # Daily range as percentage of open
        result["DailyRange_Pct"] = (result[HIGH] - result[LOW]) / (result[OPEN] + EPSILON)
        
        # Close position within daily range (0 = closed at low, 1 = closed at high)
        daily_range = result[HIGH] - result[LOW]
        result["ClosePosition"] = np.where(
            daily_range > EPSILON,
            (result[CLOSE] - result[LOW]) / daily_range,
            0.5  # When range is 0, default to middle
        )
        
        # Upper shadow ratio (how much price rejected the high)
        result["UpperShadow_Pct"] = (result[HIGH] - np.maximum(result[OPEN], result[CLOSE])) / (result[HIGH] - result[LOW] + EPSILON)
        
        # Lower shadow ratio (how much price rejected the low)
        result["LowerShadow_Pct"] = (np.minimum(result[OPEN], result[CLOSE]) - result[LOW]) / (result[HIGH] - result[LOW] + EPSILON)
        
        # Body size ratio (real body vs total range)
        result["BodySize_Pct"] = np.abs(result[CLOSE] - result[OPEN]) / (result[HIGH] - result[LOW] + EPSILON)
        
        # Gap from previous close (if we have it) - computed as open vs close ratio
        result["Open_Close_Ratio"] = result[OPEN] / (result[CLOSE] + EPSILON)
        
        # High-Low vs Close ratio (volatility indicator)
        result["HL_Close_Ratio"] = (result[HIGH] - result[LOW]) / (result[CLOSE] + EPSILON)
    
    # Volume-price relationships
    if VOLUME in result.columns and CLOSE in result.columns:
        # Dollar volume (approximate) - useful for liquidity
        result["DollarVolume"] = result[VOLUME] * result[CLOSE]
    
    # Interest rate spread features (from macro data)
    if LONG_TERM_INTEREST_RATE in result.columns:
        if IMMEDIATE_INTEREST_RATE in result.columns:
            result[IMMEDIATE_INTEREST_VOLATILITY] = (
                result[LONG_TERM_INTEREST_RATE] - 
                result[IMMEDIATE_INTEREST_RATE]
            )
        
        if SHORT_TERM_INTEREST_RATE in result.columns:
            result[SHORT_TERM_INTEREST_VOLATILITY] = (
                result[LONG_TERM_INTEREST_RATE] - 
                result[SHORT_TERM_INTEREST_RATE]
            )
    
    # Fundamental ratios (from nzx-predictor's add_engineered_features)
    result = _add_fundamental_ratios(result)
    
    return result


def _add_fundamental_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add fundamental ratio features from nzx-predictor.
    
    These ratios preserve relationships between financial metrics that
    would otherwise be lost after scaling.
    
    Args:
        df: Wide format DataFrame with fundamental columns.
    
    Returns:
        DataFrame with fundamental ratio features added.
    """
    result = df.copy()
    
    # EPS (Basic) = Net Income / Basic Average Shares
    # Note: Column names may have 'annual' prefix from the data
    net_income_col = _find_column(result, ANNUAL_NET_INCOME, "annualNetIncome")
    avg_shares_col = _find_column(result, ANNUAL_BASIC_AVG_SHARES, "annualBasicAverageShares")
    
    if net_income_col and avg_shares_col:
        result[EPS_BASIC] = result[net_income_col] / (result[avg_shares_col] + EPSILON)
    
    # Net Profit Margin = Net Income / Total Revenue
    total_revenue_col = _find_column(result, ANNUAL_TOTAL_REVENUE, "annualTotalRevenue")
    
    if net_income_col and total_revenue_col:
        result[NET_PROFIT_MARGIN] = result[net_income_col] / (result[total_revenue_col] + EPSILON)
    
    # SG&A Ratio = SG&A / Total Revenue
    sga_col = _find_column(result, ANNUAL_SGA, "annualSellingGeneralAndAdministration")
    
    if sga_col and total_revenue_col:
        result[SGA_RATIO] = result[sga_col] / (result[total_revenue_col] + EPSILON)
    
    # Depreciation Ratio = Depreciation / Total Revenue
    depreciation_col = _find_column(
        result, 
        ANNUAL_DEPRECIATION, 
        "annualDepreciationAmortizationDepletionIncomeStatement"
    )
    
    if depreciation_col and total_revenue_col:
        result[DEPRECIATION_RATIO] = result[depreciation_col] / (result[total_revenue_col] + EPSILON)
    
    # Commission Efficiency = Fees and Commission Expense / Total Revenue
    fees_col = _find_column(result, TRAILING_FEES_COMMISSION, "trailingFeesandCommissionExpense")
    
    if fees_col and total_revenue_col:
        result[COMMISSION_EFFICIENCY] = result[fees_col] / (result[total_revenue_col] + EPSILON)
    
    return result


def _find_column(df: pd.DataFrame, *candidates: str) -> str | None:
    """Find first matching column name from candidates.
    
    Useful when column names may vary slightly (e.g., with/without prefixes).
    """
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None
    
    return result
