"""
Configuration module for ML pipeline constants and paths.
Defines all configuration parameters, column names, file paths, and feature engineering settings.
"""
from src.utils.path_utils import get_skuld_root


_day = 1000 * 60 * 60 * 24
_year = _day * 365

# === LABELING & LOOKAHEAD PARAMETERS ===
THRESHOLD_PCT = 0.13  # the % price change threshold for 1 label
# Example: 0.02 = +2% price increase

LABEL_LOOKAHEAD_MILLIS = _year * 1.001  # milliseconds into the future for price movement
TEST_SPLIT_DURATION_MILLIS = _day * 30  # test split size by time
EVAL_TEST_ITERATIONS = 5  # how many iterations to run sliding window over
EVAL_TIME_PROGRESSION = _year * 1.314159  # amount of time to advance evaluation per iteration
EVAL_CLASSIFICATION_BOUNDARY = 0.5

# === FINANCIAL FEATURE COLUMNS ===
OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
VOLUME_COL = "Volume"
ANNUAL_BASIC_EPS_COL = "annualBasicEPS"
ANNUAL_DILUTED_EPS_COL = "annualDilutedEPS"
TRAILING_BASIC_EPS_COL = "trailingBasicEPS"
TRAILING_DILUTED_EPS_COL = "trailingDilutedEPS"
ANNUAL_NET_INCOME_COL = "annualNetIncome"
TRAILING_NET_INCOME_COL = "trailingNetIncome"
ANNUAL_TOTAL_REVENUE_COL = "annualTotalRevenue"
TRAILING_TOTAL_REVENUE_COL = "trailingTotalRevenue"
ANNUAL_TOTAL_UNUSUAL_ITEMS_COL = "annualTotalUnusualItems"
TRAILING_TOTAL_UNUSUAL_ITEMS_COL = "trailingTotalUnusualItems"
ANNUAL_GA_EXPENSE_COL = "annualGeneralAndAdministrativeExpense"
TRAILING_GA_EXPENSE_COL = "trailingGeneralAndAdministrativeExpense"
ANNUAL_DILUTED_AVG_SHARES_COL = "annualDilutedAverageShares"
ANNUAL_EBITDA_COL = "annualEBITDA"
TRAILING_EBITDA_COL = "trailingEBITDA"
NZL_EMP_Y15T64_T_COL = "NZL_LaborStats_EMP_Y15T64__T"
NZL_POP_Y15T64_T_COL = "NZL_LaborStats_POP_Y15T64__T"
NZL_LF_Y15T64_T_COL = "NZL_LaborStats_LF_Y15T64__T"
ANNUAL_EBIT_COL = "annualEBIT"
ANNUAL_INTEREST_EXPENSE_COL = "annualInterestExpense"
LONG_TERM_RATE_COL = "Long-term interest rates"
SHORT_TERM_RATE_COL = "Short-term interest rates"

# === COLUMN NAMES ===
TIMESTAMP_COL = "timestamp"
TICKER_COL = "ticker"
CLOSE_COL = "Close"
LABEL_COL = "label"
PRICE_COL = "Close"
PREDICTION_COL = "pred_prob"
TICKER_PREFIX = "#TICKER#"
TIMESTAMP_SCALED_COL = "scaled_timestamp"  # Added columns

# === FILE PATHS ===
_root = get_skuld_root()
# Intermediate files use Parquet for speed (10-100x faster than CSV)
TRAIN_CSV_PATH = _root / "python-ml" / "data" / "train.parquet"
TEST_CSV_PATH = _root / "python-ml" / "data" / "test.parquet"
PREPROCESSED_CSV_PATH = _root / "python-ml" / "data" / "data_preprocessed.parquet"
WIDE_CSV_PATH = _root / "python-ml" / "data" / "data_wide_imputed.parquet"

# External-facing files stay as CSV for easy inspection
MODEL_PKL_PATH = _root / "python-ml" / "data" / "model.pkl"
PREDICTION_CSV_PATH = _root / "python-ml" / "data" / "predictions.csv"  # External CSV
LONG_CSV_PATH = _root / "data" / "data_long.csv"  # Input CSV
PY_DATA_DIR_PATH = _root / "python-ml" / "data"
AGGREGATE_PREDICTIONS_CSV_PATH = _root / "python-ml" / "data" / "aggregate_predictions.csv"  # External CSV
TRADE_SIMULATION_CSV_PATH = _root / "data" / "trade_simulation.csv"  # External CSV
EVALUATION_RESULTS_CSV_PATH = _root / "data" / "evaluation_metrics.csv"  # External CSV
FINAL_PREDICTIONS_CSV_PATH = _root / "data" / "predictions.csv"  # External CSV

# === FEATURE ENGINEERING FLAGS ===
FE_ENABLE_DROP_SPARSE_COLUMNS: bool = True
# NOTE: Data scaling is now handled in post_split_preprocessing to prevent leakage.
# The following flags are deprecated and should not be used before train/test split.
FE_ENABLE_MIN_MAX_SCALE_TIME_COLUMN: bool = False  # DEPRECATED - handled post-split
FE_ENABLE_SCALE_WHOLE_DATASET: bool = False  # DEPRECATED - handled post-split
