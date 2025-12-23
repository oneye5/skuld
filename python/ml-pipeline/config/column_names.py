"""Column name constants used throughout the pipeline."""

# Raw data columns (long format)
TIMESTAMP = "timestamp"
TICKER = "ticker"
FEATURE = "feature"
VALUE = "value"

# Target/label column
TARGET = "target"

# Special feature values
CLOSE = "Close"
OPEN = "Open"
HIGH = "High"
LOW = "Low"
VOLUME = "Volume"

# Prefixes
MACRO_PREFIX = "MACRO_"
IMPUTED_PREFIX = "imputed_"
MISSING_PREFIX = "missing_"

# Generated columns
PREDICTION_PROB = "prediction_probability"
PREDICTION = "prediction"

# Time features
DAY_OF_YEAR_SIN = "day_of_year_sin"
DAY_OF_YEAR_COS = "day_of_year_cos"
DAY_OF_WEEK_SIN = "day_of_week_sin"
DAY_OF_WEEK_COS = "day_of_week_cos"
MONTH_SIN = "month_sin"
MONTH_COS = "month_cos"

# Financial data columns (from raw data)
ANNUAL_NET_INCOME = "annualNetIncome"
ANNUAL_BASIC_AVERAGE_SHARE = "annualBasicAverageShares"
ANNUAL_TOTAL_REVENUE = "annualTotalRevenue"
ANNUAL_SGA = "annualSellingGeneralAndAdministration"
ANNUAL_DEPRECIATION = "annualDepreciationIncomeStatement"
TRAILING_FEES_COMMISSION = "trailingFeesandCommissionExpense"
LONG_TERM_INTEREST_RATE = "Long-term interest rates"
IMMEDIATE_INTEREST_RATE = "Immediate interest rates- call money- interbank rate"
SHORT_TERM_INTEREST_RATE = "Short-term interest rates"

# Engineered ratio features
EPS_BASIC = "EPS_Basic"
NET_PROFIT_MARGIN = "NetProfitMargin"
SGA_RATIO = "SGA_Ratio"
DEPRECIATION_RATIO = "DepreciationRatio"
COMMISSION_EFFICIENCY = "CommissionEfficiency"
IMMEDIATE_INTEREST_VOLATILITY = "ImmediateInterestVolatility"
SHORT_TERM_INTEREST_VOLATILITY = "ShortTermInterestVolatility"