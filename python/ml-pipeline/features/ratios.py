"""Financial ratio features from nzx-predictor.

These ratios preserve relationships between features that would be lost after scaling.
Includes both OHLCV-derived features and fundamental ratios from nzx-predictor.
Also includes derived cross-asset features (yield curve, gold/oil ratio, etc.).
"""

import pandas as pd
import numpy as np

from config.columns import (
    OPEN, HIGH, LOW, CLOSE, VOLUME, TICKER, TIMESTAMP,
    DIVIDEND, TRAILING_DIV_YIELD_252,
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
    # Derived ratio feature names
    YIELD_CURVE_SPREAD,
    VOL_TERM_STRUCTURE,
    GOLD_OIL_RATIO,
    FX_ADJUSTED_RETURN,
    RELATIVE_TO_MARKET,
    DOLLAR_VOL_MARKET_SHARE,
    FEAR_RATIO,
    EARNINGS_QUALITY,
    AQUACULTURE_TREND,
    # Macro column names
    MACRO_GOLD_ADJCLOSE,
    MACRO_OIL_ADJCLOSE,
    MACRO_NZDUSD,
    MACRO_NZSE_ADJCLOSE,
    MACRO_TNX,
)
from config.settings import EPSILON


# Maximum value to prevent float overflow in ratios
MAX_RATIO_VALUE = 1e6
MIN_RATIO_VALUE = -1e6


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
    
    # Trailing dividend yield (safe, backward-looking)
    result = _add_trailing_dividend_yield(result)
    
    # Cross-asset derived features (yield curve, gold/oil, etc.)
    result = _add_derived_features(result)
    
    return result


def _add_trailing_dividend_yield(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Add trailing dividend yield feature.
    
    Computes trailing 12-month (252 trading days) dividend yield:
        TrailingDivYield = Sum(Dividends over past 252 days) / Current Price
    
    This is a safe, backward-looking feature that doesn't cause data leakage.
    The raw Dividend column (point-in-time payments) should be excluded from
    model features, but this derived yield is safe to use.
    
    Args:
        df: Wide format DataFrame with Dividend and Close columns.
        window: Lookback window in trading days (default 252 = ~1 year).
    
    Returns:
        DataFrame with TrailingDivYield_252d column added.
    """
    result = df.copy()
    
    if DIVIDEND not in result.columns or CLOSE not in result.columns:
        return result
    
    if TICKER not in result.columns:
        # No ticker column - can't compute per-stock trailing dividends
        return result
    
    # Fill NaN dividends with 0 (no dividend on that day)
    dividends = result[DIVIDEND].fillna(0)
    
    # Compute trailing sum of dividends per ticker
    # Group by ticker and compute rolling sum
    trailing_div = (
        result.assign(_div=dividends)
        .groupby(TICKER)['_div']
        .transform(lambda x: x.rolling(window=window, min_periods=1).sum())
    )
    
    # Dividend yield = trailing dividends / current price
    result[TRAILING_DIV_YIELD_252] = trailing_div / (result[CLOSE] + EPSILON)
    
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


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide two series, handling division by zero and overflow.
    
    Args:
        numerator: Numerator series.
        denominator: Denominator series.
    
    Returns:
        Result series with NaN for invalid divisions, clipped to prevent overflow.
    """
    # Add epsilon to avoid division by zero, but only where denominator is near zero
    safe_denom = np.where(
        np.abs(denominator) < EPSILON,
        np.sign(denominator + EPSILON) * EPSILON,  # Preserve sign
        denominator
    )
    
    result = numerator / safe_denom
    
    # Replace infinities with NaN
    result = result.replace([np.inf, -np.inf], np.nan)
    
    # Clip extreme values to prevent float overflow
    result = result.clip(MIN_RATIO_VALUE, MAX_RATIO_VALUE)
    
    return result


def _safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Safely compute percentage change, handling edge cases.
    
    Args:
        series: Input series.
        periods: Number of periods for change calculation.
    
    Returns:
        Percentage change with NaN for invalid values.
    """
    prev = series.shift(periods)
    result = _safe_divide(series - prev, prev)
    return result


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-asset derived features.
    
    These features combine multiple data sources to create meaningful ratios:
    - Yield curve spread (long-term vs short-term rates)
    - Volatility term structure (short-term vs long-term vol)
    - Gold/Oil ratio (macro risk indicator)
    - FX-adjusted returns
    - Relative performance vs market
    - Dollar volume market share
    - Fear ratio (aggregated Wiki fear terms)
    - Earnings quality (cash flow vs income)
    - Aquaculture trend
    
    All computations use safe division to handle edge cases.
    
    Args:
        df: Wide format DataFrame with all features.
    
    Returns:
        DataFrame with derived features added.
    """
    result = df.copy()
    
    # 1. Yield Curve Spread (long-term - short-term rates)
    result = _add_yield_curve_spread(result)
    
    # 2. Volatility Term Structure (short-term vol / long-term vol)
    result = _add_vol_term_structure(result)
    
    # 3. Gold/Oil Ratio (macro risk indicator)
    result = _add_gold_oil_ratio(result)
    
    # 4. FX-Adjusted Returns
    result = _add_fx_adjusted_return(result)
    
    # 5. Relative Performance vs Market Index
    result = _add_relative_to_market(result)
    
    # 6. Dollar Volume Market Share
    result = _add_dollar_vol_market_share(result)
    
    # 7. Fear Ratio (aggregated Wiki fear terms)
    result = _add_fear_ratio(result)
    
    # 8. Earnings Quality
    result = _add_earnings_quality(result)
    
    # 9. Aquaculture Trend
    result = _add_aquaculture_trend(result)
    
    return result


def _add_yield_curve_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Add yield curve spread: long-term rate minus short-term rate.
    
    Inverted yield curve (negative spread) often predicts recession.
    """
    result = df.copy()
    
    # Try TNX (10Y Treasury) as long-term proxy
    long_rate_col = _find_column(
        result, 
        MACRO_TNX,
        LONG_TERM_INTEREST_RATE,
        "MACRO_Long-term interest rates"
    )
    
    short_rate_col = _find_column(
        result,
        SHORT_TERM_INTEREST_RATE,
        IMMEDIATE_INTEREST_RATE,
        "MACRO_Short-term interest rates",
        "MACRO_Immediate interest rates- call money- interbank rate"
    )
    
    if long_rate_col and short_rate_col:
        spread = result[long_rate_col] - result[short_rate_col]
        # Clip to reasonable range (-10% to +10%)
        result[YIELD_CURVE_SPREAD] = spread.clip(-10, 10)
    
    return result


def _add_vol_term_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility term structure: short-term vol / long-term vol.
    
    > 1: Short-term more volatile (uncertainty/stress)
    < 1: Long-term more volatile (stable short-term)
    
    Computed per-ticker to avoid cross-contamination.
    """
    result = df.copy()
    
    vol_20_col = _find_column(result, "Vol_20", "Vol_20d")
    vol_252_col = _find_column(result, "Vol_252", "Vol_252d")
    
    if vol_20_col and vol_252_col:
        result[VOL_TERM_STRUCTURE] = _safe_divide(
            result[vol_20_col],
            result[vol_252_col]
        )
        # Clip to reasonable range
        result[VOL_TERM_STRUCTURE] = result[VOL_TERM_STRUCTURE].clip(0.1, 10)
    
    return result


def _add_gold_oil_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add gold/oil ratio: gold price / oil price.
    
    High ratio = risk-off environment (gold up, oil down)
    Low ratio = risk-on environment (oil up, growth expectations)
    """
    result = df.copy()
    
    gold_col = _find_column(result, MACRO_GOLD_ADJCLOSE, "MACRO_GC=F_Close")
    oil_col = _find_column(result, MACRO_OIL_ADJCLOSE, "MACRO_CL=F_Close")
    
    if gold_col and oil_col:
        result[GOLD_OIL_RATIO] = _safe_divide(
            result[gold_col],
            result[oil_col]
        )
        # Clip to historical reasonable range
        result[GOLD_OIL_RATIO] = result[GOLD_OIL_RATIO].clip(5, 100)
    
    return result


def _add_fx_adjusted_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add FX-adjusted return: stock return adjusted for NZD/USD movements.
    
    For NZ stocks, this measures return in USD terms, which matters for
    international investors.
    
    Computed per-ticker.
    """
    result = df.copy()
    
    fx_col = _find_column(result, MACRO_NZDUSD, "MACRO_NZDUSD=X_Close")
    
    if fx_col and CLOSE in result.columns and TICKER in result.columns:
        # Compute stock return and FX return
        result = result.sort_values([TICKER, TIMESTAMP])
        
        # Stock 1-day return per ticker
        stock_ret = result.groupby(TICKER)[CLOSE].transform(
            lambda x: x.pct_change(1)
        )
        
        # FX 1-day return (global, not per ticker)
        fx_ret = result[fx_col].pct_change(1)
        
        # FX-adjusted return = stock_ret * (1 + fx_ret) - 1
        # Approximation: stock_ret + fx_ret for small returns
        result[FX_ADJUSTED_RETURN] = stock_ret + fx_ret
        
        # Clip extreme values
        result[FX_ADJUSTED_RETURN] = result[FX_ADJUSTED_RETURN].clip(-0.5, 0.5)
    
    return result


def _add_relative_to_market(df: pd.DataFrame) -> pd.DataFrame:
    """Add relative performance: stock return minus market index return.
    
    Positive = outperforming market
    Negative = underperforming market
    
    Computed per-ticker.
    """
    result = df.copy()
    
    market_col = _find_column(
        result, 
        MACRO_NZSE_ADJCLOSE,
        "MACRO_%5ENZSE_Close",
        "MACRO_^NZSE_AdjClose"
    )
    
    if market_col and CLOSE in result.columns and TICKER in result.columns:
        result = result.sort_values([TICKER, TIMESTAMP])
        
        # Stock 1-day return per ticker
        stock_ret = result.groupby(TICKER)[CLOSE].transform(
            lambda x: x.pct_change(1)
        )
        
        # Market 1-day return (global)
        market_ret = result[market_col].pct_change(1)
        
        # Relative return = stock return - market return
        result[RELATIVE_TO_MARKET] = stock_ret - market_ret
        
        # Clip extreme values
        result[RELATIVE_TO_MARKET] = result[RELATIVE_TO_MARKET].clip(-0.5, 0.5)
    
    return result


def _add_dollar_vol_market_share(df: pd.DataFrame) -> pd.DataFrame:
    """Add dollar volume market share: stock's volume / total market volume.
    
    Measures how much of total market activity is in this stock.
    High share = heavily traded, possibly in focus
    
    Computed per-timestamp across all tickers.
    """
    result = df.copy()
    
    dollar_vol_col = _find_column(result, "DollarVolume", "DolVol")
    
    if dollar_vol_col and TIMESTAMP in result.columns:
        # Total dollar volume per timestamp
        total_vol = result.groupby(TIMESTAMP)[dollar_vol_col].transform('sum')
        
        # Market share
        result[DOLLAR_VOL_MARKET_SHARE] = _safe_divide(
            result[dollar_vol_col],
            total_vol
        )
        
        # Clip to [0, 1] range
        result[DOLLAR_VOL_MARKET_SHARE] = result[DOLLAR_VOL_MARKET_SHARE].clip(0, 1)
    
    return result


def _add_fear_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add aggregated fear ratio from Wiki fear term views.
    
    Combines multiple fear-related Wikipedia view columns into a single
    composite indicator. High values = elevated fear/uncertainty.
    """
    result = df.copy()
    
    # Look for Wiki fear columns
    fear_terms = [
        "Recession", "Unemployment", "Inflation", "Bank_run",
        "Financial_crisis", "Stock_market_crash", "Bankruptcy"
    ]
    
    fear_cols = []
    for term in fear_terms:
        col = _find_column(
            result,
            f"MACRO_{term}_Wiki_Views",
            f"MACRO_{term}_wiki_views"
        )
        if col:
            fear_cols.append(col)
    
    if fear_cols:
        # Normalize each fear column by its mean to make them comparable
        fear_values = []
        for col in fear_cols:
            col_mean = result[col].mean()
            if col_mean > EPSILON:
                normalized = result[col] / col_mean
                fear_values.append(normalized)
        
        if fear_values:
            # Average the normalized fear indicators
            fear_df = pd.concat(fear_values, axis=1)
            result[FEAR_RATIO] = fear_df.mean(axis=1)
            
            # Clip to reasonable range
            result[FEAR_RATIO] = result[FEAR_RATIO].clip(0, 10)
    
    return result


def _add_earnings_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add earnings quality: operating cash flow / net income.
    
    > 1: High quality (cash backing earnings)
    < 1: Low quality (accruals-heavy, less sustainable)
    
    Uses trailing data if available.
    """
    result = df.copy()
    
    # Look for operating cash flow
    ocf_col = _find_column(
        result,
        "trailingOperatingCashFlow",
        "annualOperatingCashFlow", 
        "trailingCashFlowFromContinuingOperatingActivities"
    )
    
    net_income_col = _find_column(
        result,
        "trailingNetIncome",
        ANNUAL_NET_INCOME,
        "annualNetIncome"
    )
    
    if ocf_col and net_income_col:
        result[EARNINGS_QUALITY] = _safe_divide(
            result[ocf_col],
            result[net_income_col]
        )
        
        # Clip to reasonable range (-5 to 5)
        result[EARNINGS_QUALITY] = result[EARNINGS_QUALITY].clip(-5, 5)
    
    return result


def _add_aquaculture_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Add NZ aquaculture production trend.
    
    Important for NZ export-oriented companies in seafood sector.
    Computes year-over-year change in production.
    """
    result = df.copy()
    
    # Look for aquaculture columns
    aqua_col = _find_column(
        result,
        "MACRO_NZ_aquaculture",
        "MACRO_NZ_Aquaculture_Production",
        "MACRO_Aquaculture_NZ"
    )
    
    if aqua_col and TIMESTAMP in result.columns:
        # Sort by timestamp
        result = result.sort_values(TIMESTAMP)
        
        # Compute 252-day (1 year) rolling change
        # Use pct_change with 252 periods for YoY trend
        result[AQUACULTURE_TREND] = result[aqua_col].pct_change(252)
        
        # Clip to reasonable range
        result[AQUACULTURE_TREND] = result[AQUACULTURE_TREND].clip(-1, 1)
    
    return result
