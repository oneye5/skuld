"""Config package - centralized configuration for the ML pipeline."""

from config.settings import (
    # Target settings
    LOOKAHEAD_DAYS,
    GAIN_THRESHOLD_PCT,
    PREDICTION_THRESHOLD,
    # Rolling window settings
    NUM_ROLLING_WINDOWS,
    ROLLING_WINDOW_MOVEMENT_YEARS,
    TEST_PERIOD_YEARS,
    # Trading simulation settings
    INITIAL_CAPITAL,
    TRANSACTION_COST_PCT,
    MAX_POSITION_SIZE_PCT,
    # Constants
    MS_PER_DAY,
    YEAR_2000_MS,
    EPSILON,
    # Functions
    create_model,
    get_config_dict,
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
    # Settings
    "LOOKAHEAD_DAYS",
    "GAIN_THRESHOLD_PCT",
    "PREDICTION_THRESHOLD",
    "NUM_ROLLING_WINDOWS",
    "ROLLING_WINDOW_MOVEMENT_YEARS",
    "TEST_PERIOD_YEARS",
    "INITIAL_CAPITAL",
    "TRANSACTION_COST_PCT",
    "MAX_POSITION_SIZE_PCT",
    "MS_PER_DAY",
    "YEAR_2000_MS",
    "EPSILON",
    "create_model",
    "get_config_dict",
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
    "MACRO_PREFIX",
    # Paths
    "DATA_DIR",
    "DATA_LONG_CSV",
    "OUTPUT_DIR",
    "get_run_dir",
    "ensure_output_dirs",
]
