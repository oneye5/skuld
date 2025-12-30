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
OPEN: str = "Open"
HIGH: str = "High"
LOW: str = "Low"
VOLUME: str = "Volume"


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
