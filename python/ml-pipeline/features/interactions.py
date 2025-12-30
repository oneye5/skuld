"""Feature interaction module.

Creates interaction features between different feature types:
- Momentum × Volatility (already in technical.py)
- Interest Rate × Stock Features (here)
- Macro × Micro interactions

These interactions capture regime-dependent effects:
- High-debt stocks are more sensitive to interest rate changes
- Momentum strategies may work differently in different rate environments
"""

import pandas as pd
import numpy as np

from config.columns import (
    TIMESTAMP, TICKER,
    LONG_TERM_INTEREST_RATE,
    IMMEDIATE_INTEREST_RATE,
    SHORT_TERM_INTEREST_RATE,
)
from config.settings import EPSILON

# Fundamental columns for debt proxy
ANNUAL_INTEREST_EXPENSE = "annualInterestExpense"
ANNUAL_TOTAL_REVENUE = "annualTotalRevenue"
ANNUAL_EBITDA = "annualEBITDA"


def add_interest_rate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add interest rate × stock feature interactions.
    
    Creates interactions between macro interest rate data and stock-specific
    features. These capture regime-dependent behavior:
    - Rate sensitivity of momentum
    - Rate sensitivity of volatility
    - Rate cycle effects
    
    Args:
        df: Wide format DataFrame with both MACRO_* interest rate columns
            and stock-specific features.
    
    Returns:
        DataFrame with IR_* interaction features added.
    """
    result = df.copy()
    
    # Identify which interest rate columns are available
    ir_cols = []
    if LONG_TERM_INTEREST_RATE in result.columns:
        ir_cols.append(("LT_IR", LONG_TERM_INTEREST_RATE))
    if SHORT_TERM_INTEREST_RATE in result.columns:
        ir_cols.append(("ST_IR", SHORT_TERM_INTEREST_RATE))
    if IMMEDIATE_INTEREST_RATE in result.columns:
        ir_cols.append(("IM_IR", IMMEDIATE_INTEREST_RATE))
    
    if not ir_cols:
        # No interest rate data available
        return df
    
    # Use the first available interest rate (prefer long-term)
    ir_prefix, ir_col = ir_cols[0]
    ir_data = result[ir_col].fillna(0)
    
    # --- Interest Rate × Momentum ---
    # Momentum may be more/less effective in different rate environments
    if "ROC_252" in result.columns:
        result[f"IR_x_Mom252"] = ir_data * result["ROC_252"]
    
    if "ROC_10" in result.columns:
        result[f"IR_x_Mom10"] = ir_data * result["ROC_10"]
    
    # --- Interest Rate × Volatility ---
    # High rates + high volatility = risk regime
    if "Vol_20" in result.columns:
        result[f"IR_x_Vol20"] = ir_data * result["Vol_20"]
    
    if "Vol_252" in result.columns:
        result[f"IR_x_Vol252"] = ir_data * result["Vol_252"]
    
    # --- Interest Rate × RSI ---
    # RSI signal strength in different rate regimes
    if "RSI_14" in result.columns:
        result[f"IR_x_RSI"] = ir_data * result["RSI_14"]
    
    # --- Interest Rate × Distance from MA ---
    # Trend strength in different rate environments
    if "Dist_MA_200" in result.columns:
        result[f"IR_x_Trend"] = ir_data * result["Dist_MA_200"]
    
    # --- Interest Rate Change Features ---
    # Rate changes matter more than levels for stock reactions
    if TIMESTAMP in result.columns:
        # Calculate interest rate change (per ticker-timestamp pair not ideal,
        # but works since IR is same across all tickers at each timestamp)
        ir_change = result.groupby(TICKER)[ir_col].diff()
        result["IR_Change"] = ir_change.fillna(0)
        
        # Stock momentum × rate change (rate hikes hurt momentum stocks)
        if "ROC_252" in result.columns:
            result["IR_Change_x_Mom252"] = result["IR_Change"] * result["ROC_252"]
    
    return result


def add_debt_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add debt proxy features derived from interest expense.
    
    Since balance sheet data (Total Debt, Equity) is not available,
    we use interest expense as a proxy for debt level. Companies with
    higher interest expense relative to revenue/EBITDA have more debt.
    
    Features created:
    - InterestBurden: Interest expense / Revenue (higher = more debt)
    - InterestCoverage: EBITDA / Interest expense (lower = more leveraged)
    - IR_x_InterestBurden: Interest rate × debt proxy interaction
    
    Args:
        df: Wide format DataFrame with fundamental columns.
    
    Returns:
        DataFrame with debt proxy features added.
    """
    result = df.copy()
    
    # --- Interest Burden (proxy for debt/revenue) ---
    if ANNUAL_INTEREST_EXPENSE in result.columns and ANNUAL_TOTAL_REVENUE in result.columns:
        interest_exp = result[ANNUAL_INTEREST_EXPENSE].fillna(0).abs()
        revenue = result[ANNUAL_TOTAL_REVENUE].fillna(0).abs()
        
        # Interest expense as % of revenue (higher = more debt burden)
        result["InterestBurden"] = interest_exp / (revenue + EPSILON)
        # Cap at reasonable value (interest > 50% of revenue is extreme)
        result["InterestBurden"] = result["InterestBurden"].clip(upper=0.5)
    
    # --- Interest Coverage Ratio (EBITDA / Interest Expense) ---
    if ANNUAL_INTEREST_EXPENSE in result.columns and ANNUAL_EBITDA in result.columns:
        interest_exp = result[ANNUAL_INTEREST_EXPENSE].fillna(0).abs()
        ebitda = result[ANNUAL_EBITDA].fillna(0)
        
        # Interest coverage: higher = less leveraged, more able to service debt
        # Use inverse so that higher = more debt risk (consistent with InterestBurden)
        coverage = ebitda / (interest_exp + EPSILON)
        # Clip and invert: low coverage (< 2) is risky
        coverage_clipped = coverage.clip(lower=0.1, upper=20)
        result["InterestCoverageInv"] = 1.0 / coverage_clipped
    
    # --- Interest Rate × Debt Proxy Interaction ---
    # High-debt companies are more sensitive to interest rate changes
    ir_col = None
    if LONG_TERM_INTEREST_RATE in result.columns:
        ir_col = LONG_TERM_INTEREST_RATE
    elif SHORT_TERM_INTEREST_RATE in result.columns:
        ir_col = SHORT_TERM_INTEREST_RATE
    
    if ir_col is not None:
        ir_data = result[ir_col].fillna(0)
        
        if "InterestBurden" in result.columns:
            # Rate × debt burden: high rates hurt high-debt companies more
            result["IR_x_DebtBurden"] = ir_data * result["InterestBurden"]
        
        if "InterestCoverageInv" in result.columns:
            # Rate × low coverage: rate hikes hurt companies with low coverage
            result["IR_x_LowCoverage"] = ir_data * result["InterestCoverageInv"]
    
    return result


def add_cross_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features derived from multiple macro indicators.
    
    Creates features combining different macro indicators:
    - Yield curve slope (long-term - short-term rate)
    - Rate spread features
    
    Args:
        df: Wide format DataFrame with MACRO_* columns.
    
    Returns:
        DataFrame with cross-macro features added.
    """
    result = df.copy()
    
    # --- Yield Curve Slope ---
    # Long-term rate - Short-term rate
    # Inverted yield curve (negative slope) predicts recessions
    if LONG_TERM_INTEREST_RATE in result.columns and SHORT_TERM_INTEREST_RATE in result.columns:
        result["YieldCurveSlope"] = (
            result[LONG_TERM_INTEREST_RATE].fillna(0) - 
            result[SHORT_TERM_INTEREST_RATE].fillna(0)
        )
        
        # Yield curve × momentum interaction
        if "ROC_252" in result.columns:
            result["YieldSlope_x_Mom252"] = result["YieldCurveSlope"] * result["ROC_252"]
    
    # --- Rate Spread (Long-term - Immediate) ---
    if LONG_TERM_INTEREST_RATE in result.columns and IMMEDIATE_INTEREST_RATE in result.columns:
        result["LT_IM_Spread"] = (
            result[LONG_TERM_INTEREST_RATE].fillna(0) - 
            result[IMMEDIATE_INTEREST_RATE].fillna(0)
        )
    
    return result


def add_all_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add all interaction features.
    
    This is a convenience function that applies all interaction feature
    modules in the correct order.
    
    Args:
        df: Wide format DataFrame.
    
    Returns:
        DataFrame with all interaction features added.
    """
    # Note: Momentum × Volatility interactions are already added in technical.py
    # via _add_interaction_features(), so we don't duplicate them here.
    
    # Debt proxy features (must come before IR interactions that use them)
    df = add_debt_proxy_features(df)
    
    # Interest rate × stock feature interactions
    df = add_interest_rate_interactions(df)
    
    # Cross-macro features
    df = add_cross_macro_features(df)
    
    return df
