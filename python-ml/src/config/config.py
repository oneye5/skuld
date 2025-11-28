from src.utils.path_utils import get_skuld_root
_day = 1000 * 60 * 60 * 24
_year = _day * 365

THRESHOLD_PCT = 0.03 # the % price change threshold for 1 label
# Example: 0.02 = +2% price increase

LABEL_LOOKAHEAD_MILLIS = _year # milliseconds into the future for price movement
TEST_SPLIT_DURATION_MILLIS = _day * 1.1 # test split size by time
EVAL_TEST_ITERATIONS = 10 # how many iterations to run sliding window over
EVAL_TIME_PROGRESSION = _year # amount of time to advance evaluation per iteration
EVAL_CLASSIFICATION_BOUNDARY = 0.7

# Column names
TIMESTAMP_COL = "timestamp"
TICKER_COL = "ticker"
CLOSE_COL = "Close"
LABEL_COL = "label"
PRICE_COL = "Close"
PREDICTION_COL = "pred_prob"
TICKER_PREFIX = "#TICKER#"
# Added columns
TIMESTAMP_SCALED_COL = "scaled_timestamp"

# File paths
_root = get_skuld_root()
TRAIN_CSV_PATH = _root / "python-ml" / "data" / "train.csv"
TEST_CSV_PATH = _root / "python-ml" / "data" / "test.csv"
PREPROCESSED_CSV_PATH = _root / "python-ml" / "data" / "data_preprocessed.csv"
MODEL_PKL_PATH = _root / "python-ml" / "data" / "model.pkl"
PREDICTION_CSV_PATH = _root / "python-ml" / "data" / "predictions.csv"
LONG_CSV_PATH = _root / "data" / "data_long.csv"
WIDE_CSV_PATH = _root / "python-ml" / "data" / "data_wide_imputed.csv"
PY_DATA_DIR_PATH = _root / "python-ml" / "data"
AGGREGATE_PREDICTIONS_CSV_PATH = _root / "python-ml" / "data" / "aggregate_predictions.csv"
TRADE_SIMULATION_CSV_PATH = _root / "data" / "trade_simulation.csv"
EVALUATION_RESULTS_CSV_PATH = _root / "data" / "evaluation_metrics.csv"
FINAL_PREDICTIONS_CSV_PATH = _root / "data" / "predictions.csv"

# Feature engineering
FE_ENABLE_DROP_SPARSE_COLUMNS: bool = False
FE_ENABLE_MIN_MAX_SCALE_TIME_COLUMN: bool = False
FE_ENABLE_SCALE_WHOLE_DATASET: bool = False
