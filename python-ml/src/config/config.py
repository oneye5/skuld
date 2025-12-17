"""
Configuration module for ML pipeline constants and paths.
Defines all configuration parameters, column names, file paths, and feature engineering settings.
"""
from src.utils.path_utils import get_skuld_root

_day = 1000 * 60 * 60 * 24
_year = _day * 365

# === LABELING & LOOKAHEAD PARAMETERS ===
THRESHOLD_PCT = 0.01  # the % price change threshold for 1 label
# Example: 0.02 = +2% price increase

LABEL_LOOKAHEAD_MILLIS = _year  # milliseconds into the future for price movement
TEST_SPLIT_DURATION_MILLIS = _day * 1.1  # test split size by time
EVAL_TEST_ITERATIONS = 5  # how many iterations to run sliding window over
EVAL_TIME_PROGRESSION = _year * 1.314159  # amount of time to advance evaluation per iteration
EVAL_CLASSIFICATION_BOUNDARY = 0.75

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
