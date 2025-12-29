"""Config package - centralized configuration for the ML pipeline."""

from config.settings import (
    # Enums
    ReturnType,
    # Target settings (ranking)
    FORWARD_RETURN_DAYS,
    RETURN_TYPE,
    WINSORIZE_LIMITS,
    # Rolling window settings
    NUM_ROLLING_WINDOWS,
    ROLLING_WINDOW_MOVEMENT_YEARS,
    TEST_PERIOD_YEARS,
    # Ranking model settings
    RANKER_N_ESTIMATORS,
    RANKER_LEARNING_RATE,
    RANKER_NUM_LEAVES,
    RANKER_MAX_DEPTH,
    RANKER_MIN_CHILD_SAMPLES,
    RANKER_SUBSAMPLE,
    RANKER_COLSAMPLE_BYTREE,
    RANKER_EVAL_AT,
    MIN_STOCKS_PER_TIMESTAMP,
    # Portfolio settings
    PORTFOLIO_TOP_N,
    PORTFOLIO_BOTTOM_N,
    TRANSACTION_COST_BPS,
    LONG_ONLY,
    INITIAL_CAPITAL,
    # Evaluation settings
    MIN_STOCKS_FOR_IC,
    TOP_N_FOR_HIT_RATE,
    PERIODS_PER_YEAR,
    # Constants
    MS_PER_DAY,
    YEAR_2000_MS,
    EPSILON,
    CLIP_THRESHOLD,
)

from config.columns import (
    # Raw data columns
    TIMESTAMP,
    TICKER,
    FEATURE,
    VALUE,
    # OHLCV
    CLOSE,
    OPEN,
    HIGH,
    LOW,
    VOLUME,
    # Target
    TARGET,
    # Generated columns
    PREDICTION_PROB,
    PREDICTION,
    # Prefixes
    MACRO_PREFIX,
)

from config.paths import (
    DATA_DIR,
    DATA_LONG_CSV,
    OUTPUT_DIR,
    get_run_dir,
    ensure_output_dirs,
)

__all__ = [
    # Enums
    "ReturnType",
    # Target settings (ranking)
    "FORWARD_RETURN_DAYS",
    "RETURN_TYPE",
    "WINSORIZE_LIMITS",
    # Rolling window settings
    "NUM_ROLLING_WINDOWS",
    "ROLLING_WINDOW_MOVEMENT_YEARS",
    "TEST_PERIOD_YEARS",
    # Ranking model settings
    "RANKER_N_ESTIMATORS",
    "RANKER_LEARNING_RATE",
    "RANKER_NUM_LEAVES",
    "RANKER_MAX_DEPTH",
    "RANKER_MIN_CHILD_SAMPLES",
    "RANKER_SUBSAMPLE",
    "RANKER_COLSAMPLE_BYTREE",
    "RANKER_EVAL_AT",
    "MIN_STOCKS_PER_TIMESTAMP",
    # Portfolio settings
    "PORTFOLIO_TOP_N",
    "PORTFOLIO_BOTTOM_N",
    "TRANSACTION_COST_BPS",
    "LONG_ONLY",
    "INITIAL_CAPITAL",
    # Evaluation settings
    "MIN_STOCKS_FOR_IC",
    "TOP_N_FOR_HIT_RATE",
    "PERIODS_PER_YEAR",
    # Constants
    "MS_PER_DAY",
    "YEAR_2000_MS",
    "EPSILON",
    "CLIP_THRESHOLD",
    # Columns
    "TIMESTAMP",
    "TICKER",
    "FEATURE",
    "VALUE",
    "CLOSE",
    "OPEN",
    "HIGH",
    "LOW",
    "VOLUME",
    "TARGET",
    "PREDICTION_PROB",
    "PREDICTION",
    "MACRO_PREFIX",
    # Paths
    "DATA_DIR",
    "DATA_LONG_CSV",
    "OUTPUT_DIR",
    "get_run_dir",
    "ensure_output_dirs",
]
