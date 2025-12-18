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