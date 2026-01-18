"""Column name constants used throughout the pipeline.

All column names should be defined here. No string literals elsewhere.
"""

# =============================================================================
# RAW DATA COLUMNS (long format)
# =============================================================================
TIMESTAMP: str = "timestamp"
TICKER: str = "ticker"
FEATURE: str = "feature"
VALUE: str = "value"


# =============================================================================
# OHLCV COLUMNS
# =============================================================================
CLOSE: str = "Close"
ADJCLOSE: str = "AdjClose"
OPEN: str = "Open"
HIGH: str = "High"
LOW: str = "Low"
VOLUME: str = "Volume"
DIVIDEND: str = "Dividend"
SPLIT: str = "Split"


# =============================================================================
# ENGINEERED DIVIDEND FEATURES
# =============================================================================
TRAILING_DIV_YIELD_252: str = "TrailingDivYield_252d"
"""Trailing 12-month dividend yield = sum of dividends over past 252 days / current price."""


# =============================================================================
# TARGET / LABELS
# =============================================================================
TARGET: str = "target"


# =============================================================================
# PREDICTION OUTPUT
# =============================================================================
PREDICTION_PROB: str = "prediction_probability"
PREDICTION: str = "prediction"


# =============================================================================
# PREFIXES
# =============================================================================
MACRO_PREFIX: str = "MACRO_"
"""Prefix added to features with empty ticker (global/macro features)."""


# =============================================================================
# FUNDAMENTAL DATA COLUMNS (from data)
# These are the actual column names in the wide format after long_to_wide
# =============================================================================
# Annual fundamentals (names from data_long.csv feature column)
ANNUAL_NET_INCOME: str = "annualNetIncome"
ANNUAL_BASIC_AVG_SHARES: str = "annualBasicAverageShares"
ANNUAL_TOTAL_REVENUE: str = "annualTotalRevenue"
ANNUAL_SGA: str = "annualSellingGeneralAndAdministration"
ANNUAL_DEPRECIATION: str = "annualDepreciationAmortizationDepletionIncomeStatement"

# Trailing fundamentals
TRAILING_FEES_COMMISSION: str = "trailingFeesandCommissionExpense"

# Interest rates (macro)
LONG_TERM_INTEREST_RATE: str = "MACRO_Long-term interest rates"
IMMEDIATE_INTEREST_RATE: str = "MACRO_Immediate interest rates- call money- interbank rate"
SHORT_TERM_INTEREST_RATE: str = "MACRO_Short-term interest rates"


# =============================================================================
# ENGINEERED FEATURE NAMES
# =============================================================================
EPS_BASIC: str = "EPS_Basic"
NET_PROFIT_MARGIN: str = "NetProfitMargin"
SGA_RATIO: str = "SGA_Ratio"
DEPRECIATION_RATIO: str = "DepreciationRatio"
COMMISSION_EFFICIENCY: str = "CommissionEfficiency"
IMMEDIATE_INTEREST_VOLATILITY: str = "ImmediateInterestVolatility"
SHORT_TERM_INTEREST_VOLATILITY: str = "ShortTermInterestVolatility"


# =============================================================================
# DERIVED RATIO FEATURES
# =============================================================================
YIELD_CURVE_SPREAD: str = "YieldCurveSpread"
"""Long-term interest rate minus short-term interest rate."""

VOL_TERM_STRUCTURE: str = "VolTermStructure"
"""Short-term volatility (20d) divided by long-term volatility (252d)."""

GOLD_OIL_RATIO: str = "GoldOilRatio"
"""Gold price divided by oil price - macro risk indicator."""

FX_ADJUSTED_RETURN: str = "FX_Adjusted_Return"
"""Stock return adjusted for NZD/USD currency movements."""

RELATIVE_TO_MARKET: str = "RelativeToMarket"
"""Stock return relative to market index return."""

DOLLAR_VOL_MARKET_SHARE: str = "DollarVolMarketShare"
"""Stock's dollar volume as percentage of total market volume."""

FEAR_RATIO: str = "FearRatio"
"""Aggregated fear indicator from Wiki fear terms."""

EARNINGS_QUALITY: str = "EarningsQuality"
"""Operating cash flow relative to net income."""

AQUACULTURE_TREND: str = "AquacultureTrend"
"""NZ aquaculture production trend."""


# =============================================================================
# MACRO COLUMN NAMES (for derived features)
# =============================================================================
MACRO_GOLD_ADJCLOSE: str = "MACRO_GC=F_AdjClose"
MACRO_OIL_ADJCLOSE: str = "MACRO_CL=F_AdjClose"
MACRO_NZDUSD: str = "MACRO_NZDUSD=X_AdjClose"
MACRO_NZSE_ADJCLOSE: str = "MACRO_%5ENZSE_AdjClose"
MACRO_TNX: str = "MACRO_%5ETNX_AdjClose"
