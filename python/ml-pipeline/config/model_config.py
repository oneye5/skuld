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
MAX_POSITION_SIZE_PCT = 5.0  # Maximum % of capital per position

# XGBoost default configuration
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
}

# Data processing
MS_PER_DAY = 86_400_000  # Milliseconds per day (timestamps are in ms)

# Define ensemble model initialization function
def initialize_model(params: dict | None = None):
    """Initialize an ensemble VotingClassifier model."""
    from sklearn.ensemble import VotingClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    """
    estimators = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('et', ExtraTreesClassifier(random_state=42)),
        ('hgb', HistGradientBoostingClassifier(random_state=42)),
        ('xgb', XGBClassifier(random_state=42)),
        ('lgbm', LGBMClassifier(random_state=42, verbose=-1)),
    ]
    return VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)  # Set n_jobs=1 to avoid multiprocessing issues on Windows
    """
    return XGBClassifier(random_state=42, **(params or XGBOOST_PARAMS))