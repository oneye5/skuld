"""Model and pipeline configuration constants."""

# Target variable configuration
LOOKAHEAD_DAYS = 365  # How far into the future to predict
GAIN_THRESHOLD_PCT = 2.0  # Minimum % gain for positive class

# Rolling window configuration
NUM_ROLLING_WINDOWS = 5
ROLLING_WINDOW_MOVEMENT_YEARS = 1.3333  # How far to move window back in time

# Trading simulation configuration
PREDICTION_THRESHOLD = 0.7  # Minimum probability to trigger buy
INITIAL_CAPITAL = 100_000.0
TRANSACTION_COST_PCT = 0.1  # Cost per trade (buy or sell)
RISK_FREE_RATE = 0.0  # For Sharpe ratio calculation

# XGBoost default configuration
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
    "n_jobs": -1,
}

# Data processing
MS_PER_DAY = 86_400_000  # Milliseconds per day (timestamps are in ms)
